import pandas as pd
import numpy as np
import os
import time as tm
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from dotenv import load_dotenv

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

load_dotenv()

Finnhub_API_Key = os.getenv("FINNHUB_API_KEY", "")
SEQUENCE_LENGTH = 20
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 50

FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close',
    'MA_20', 'Volatility_20', 'RSI', 'Obv', 'sentiment',
    'MACD_Crossover', 'Volume_5'
]


# ── Model ────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # last time step only
        return torch.sigmoid(self.fc(out))


class TradingLoss(nn.Module):
    """
    Combined loss with two components:

    1. Profit weight — wrong predictions on large price moves cost more than
       wrong predictions on small ones, directly aligning training with P&L.

    2. Anti-lag penalty — technical indicators (MA, RSI, MACD) are inherently
       backward-looking, which causes models to follow trends instead of
       anticipating reversals. When the model predicts the *same direction as
       yesterday* AND is wrong, the penalty is multiplied by lag_penalty.
       This pushes the model to be skeptical of pure trend-following signals.
    """
    def __init__(self, lag_penalty: float = 1.5):
        super().__init__()
        self.lag_penalty = lag_penalty

    def forward(self, predictions, targets, price_changes, prev_directions):
        bce = F.binary_cross_entropy(predictions.squeeze(), targets, reduction='none')

        # Weight by magnitude of the upcoming price move
        profit_weights = price_changes.abs() + 1.0

        # Extra weight on trend-following errors (the lag penalty)
        pred_binary = (predictions.squeeze() >= 0.5).float()
        is_trend_following = (pred_binary == prev_directions).float()
        is_wrong = (pred_binary != targets).float()
        lag_weights = 1.0 + self.lag_penalty * is_trend_following * is_wrong

        return (bce * profit_weights * lag_weights).mean()


# ── Data pipeline ─────────────────────────────────────────────────────────────

def fetch_historical_data(symbol, days=400):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1d")
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Timestamp'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
        return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def add_features(data):
    try:
        data['Previous_Close'] = data['Close'].shift(1)
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']
        data['Volume_5'] = data['Volume'].rolling(window=5).mean()
        data['Percent_Change'] = data['Close'].pct_change()

        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['Obv'] = obv

        data['12_EMAs'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['26_EMAs'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['12_EMAs'] - data['26_EMAs']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Crossover'] = (data['MACD'] > data['Signal_Line']).astype(int)

        data['Next_Close'] = data['Close'].shift(-1)
        data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)
        data['Price_Change_Pct'] = data['Close'].pct_change().shift(-1)  # next-day % move

        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()


def fetch_finnhub_news(symbol, finnhub_api_key, days=400):
    today = datetime.today()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    url = (
        f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
        f"&from={from_date}&to={to_date}&token={finnhub_api_key}"
    )
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        news_df = pd.DataFrame(response.json())
        if news_df.empty:
            return None
        news_df['Timestamp'] = pd.to_datetime(news_df['datetime'], unit='s')
        news_df['Timestamp'] = (
            news_df['Timestamp']
            .dt.floor('D')
            .dt.tz_localize('UTC')
            .dt.tz_convert('America/New_York')
            .dt.normalize()
            .dt.tz_localize(None)
        )
        return news_df[['Timestamp', 'headline']]
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return None


def add_sentiment(data, symbol):
    if not Finnhub_API_Key:
        data['sentiment'] = 0.0
        return data
    news_df = fetch_finnhub_news(symbol, Finnhub_API_Key)
    if news_df is None:
        data['sentiment'] = 0.0
        return data
    analyzer = SentimentIntensityAnalyzer()
    news_df['sentiment'] = news_df['headline'].apply(
        lambda t: analyzer.polarity_scores(t)['compound']
    )
    sentiment_by_date = news_df.groupby('Timestamp')['sentiment'].mean().reset_index()
    merged = pd.merge(data, sentiment_by_date, how='left', on='Timestamp')
    merged['sentiment'] = merged['sentiment'].fillna(0.0)
    tm.sleep(1)
    return merged


def calculate_sharpe(predictions, close_prices, risk_free_rate=0.0):
    """Annualized Sharpe ratio of the model's strategy on the test set."""
    returns = []
    for i in range(len(predictions) - 1):
        cur, nxt = close_prices[i], close_prices[i + 1]
        if cur == 0:
            continue
        ret = (nxt - cur) / cur if predictions[i] == 1 else (cur - nxt) / cur
        returns.append(ret)
    if not returns:
        return 0.0
    r = np.array(returns)
    std = r.std()
    if std == 0:
        return 0.0
    return float((r.mean() - risk_free_rate / 252) / std * np.sqrt(252))


def create_sequences(X, y, price_changes, prev_dirs, seq_length=SEQUENCE_LENGTH):
    Xs, ys, pcs, pds = [], [], [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
        pcs.append(price_changes[i + seq_length])
        pds.append(prev_dirs[i + seq_length])
    return np.array(Xs), np.array(ys), np.array(pcs), np.array(pds)


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_gru(symbol, model_dir="Models_GRU"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_gru.pt")
    scaler_path = os.path.join(model_dir, f"{symbol}_scaler.pkl")

    print(f"\n{'='*50}")
    print(f"Processing {symbol}")

    bars_df = fetch_historical_data(symbol)
    if bars_df.empty:
        print(f"No data for {symbol}, skipping.")
        return None

    data = add_features(bars_df)
    if data.empty or len(data) < TEST_SIZE + SEQUENCE_LENGTH + 20:
        print(f"Not enough data for {symbol}, skipping.")
        return None

    data = add_sentiment(data, symbol)

    # Ensure all features present; fill any missing with 0
    for col in FEATURES:
        if col not in data.columns:
            data[col] = 0.0

    X = data[FEATURES].values
    y = data['Direction'].values.astype(np.float32)
    price_changes = data['Price_Change_Pct'].values.astype(np.float32)
    # Previous day's direction: what the market actually did yesterday
    prev_dirs = np.concatenate([[0.0], y[:-1]]).astype(np.float32)

    import pickle
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_all_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    X_seq, y_seq, pc_seq, pd_seq = create_sequences(X_all_scaled, y, price_changes, prev_dirs)

    # Train / val / test split — all chronological, no shuffling
    n_test = TEST_SIZE
    X_train_full = X_seq[:-n_test]
    y_train_full = y_seq[:-n_test]
    pc_train_full = pc_seq[:-n_test]
    pd_train_full = pd_seq[:-n_test]
    X_test = X_seq[-n_test:]
    y_test = y_seq[-n_test:]

    # Carve validation from the end of the training window
    n_val = max(20, len(X_train_full) // 5)
    X_train = X_train_full[:-n_val]
    y_train = y_train_full[:-n_val]
    pc_train = pc_train_full[:-n_val]
    pd_train = pd_train_full[:-n_val]
    X_val = X_train_full[-n_val:]
    y_val = y_train_full[-n_val:]
    pc_val = pc_train_full[-n_val:]
    pd_val = pd_train_full[-n_val:]

    def to_tensors(*arrays):
        return [torch.tensor(a, dtype=torch.float32).to(device) for a in arrays]

    loader = DataLoader(
        TensorDataset(*to_tensors(X_train, y_train, pc_train, pd_train)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    val_tensors = to_tensors(X_val, y_val, pc_val, pd_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUModel(input_size=len(FEATURES)).to(device)
    criterion = TradingLoss(lag_penalty=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded existing weights for {symbol}.")

    # Early stopping
    PATIENCE = 7
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, pc_b, pd_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b, pc_b, pd_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensors[0]), *val_tensors[1:]).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.4f})")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Train: {epoch_loss/len(loader):.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        raw_probs = model(X_test_t).squeeze().cpu().numpy()

    predictions = (raw_probs >= 0.5).astype(int)
    close_prices_test = data['Close'].values[-n_test:]

    accuracy        = accuracy_score(y_test, predictions)
    precision       = precision_score(y_test, predictions, zero_division=0)
    recall          = recall_score(y_test, predictions, zero_division=0)
    f1              = f1_score(y_test, predictions, zero_division=0)
    try:
        auc = roc_auc_score(y_test, raw_probs)
    except ValueError:
        auc = 0.0
    sharpe          = calculate_sharpe(predictions, close_prices_test)
    last_10_accuracy = accuracy_score(y_test[-10:], predictions[-10:])
    conf_matrix     = confusion_matrix(y_test, predictions)

    # Next-bar prediction using the most recent sequence
    with torch.no_grad():
        last_seq = torch.tensor(X_seq[-1:], dtype=torch.float32).to(device)
        next_prob = model(last_seq).squeeze().item()
    next_direction = "UP" if next_prob >= 0.5 else "DOWN"

    print(
        f"Acc: {accuracy:.2f} | P: {precision:.2f} | R: {recall:.2f} | "
        f"F1: {f1:.2f} | AUC: {auc:.2f} | Sharpe: {sharpe:.2f} | "
        f"Next: {next_direction} ({next_prob:.2f})"
    )
    print(f"Confusion matrix:\n{conf_matrix}")

    return {
        "symbol":           symbol,
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "auc":              auc,
        "sharpe":           sharpe,
        "last_10_accuracy": last_10_accuracy,
        "next_direction":   next_direction,
        "next_prob":        next_prob,
        "predictions":      predictions,
        "close_prices_test": close_prices_test,
        "y_test":           y_test,
        "raw_probs":        raw_probs,
        "model":            model,
        "scaler":           scaler,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

try:
    sp500_symbols = list(
        pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol']
    )
except Exception:
    sp500_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
    print("Could not fetch S&P 500 list, using default symbols.")

model_dir = "Models_GRU"


if __name__ == "__main__":
    all_results = []

    for symbol in sp500_symbols:
        try:
            result = train_gru(symbol, model_dir=model_dir)
            if result is None:
                continue

            all_results.append({
                "Symbol":        result["symbol"],
                "Accuracy_%":    round(result["accuracy"]   * 100, 2),
                "Precision_%":   round(result["precision"]  * 100, 2),
                "Recall_%":      round(result["recall"]     * 100, 2),
                "F1_%":          round(result["f1"]         * 100, 2),
                "AUC":           round(result["auc"],               4),
                "Sharpe":        round(result["sharpe"],            3),
                "Accuracy_10_%": round(result["last_10_accuracy"] * 100, 2),
                "Next_Direction": result["next_direction"],
                "Next_Prob":     round(result["next_prob"],        4),
            })

            output_file = "GRU_results.xlsx"
            try:
                out_df = pd.DataFrame(all_results)
                if os.path.exists(output_file):
                    existing = pd.read_excel(output_file)
                    out_df = pd.concat([existing, out_df], ignore_index=True)
                    out_df = out_df.drop_duplicates(subset=["Symbol"], keep="last")
                out_df.to_excel(output_file, index=False)
            except Exception as e:
                print(f"Error saving results: {e}")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
