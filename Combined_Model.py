"""
Sequential Pipeline: GRU → SGD → RL

Stage 1 (GRU):  Learns temporal patterns from price sequences.
                Outputs a next-day probability (gru_prob) and direction signal.

Stage 2 (SGD):  Uses original technical features + gru_prob as an additional input.
                The GRU signal gives the SGD information about sequential context
                it couldn't derive from a single-row view.

Stage 3 (RL):   The PPO agent's observation space is expanded to include both
                gru_prob and sgd_conf, so the agent can learn when to trust
                each model's signal and size positions accordingly.
"""

import os
import pickle
import time as tm
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random

load_dotenv()

Finnhub_API_Key = os.getenv("FINNHUB_API_KEY", "")
PORTFOLIO_SIZE = float(os.getenv("PORTFOLIO_SIZE", "100000"))

# ── Hyperparameters ───────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 20
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_LR = 0.001
GRU_EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 50

BASE_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close',
    'MA_20', 'Volatility_20', 'RSI', 'Obv', 'sentiment',
    'MACD_Crossover', 'Volume_5'
]
# Features seen by SGD and RL include GRU output
EXTENDED_FEATURES = BASE_FEATURES + ['gru_prob']
# RL state additionally sees SGD confidence
RL_FEATURES = EXTENDED_FEATURES + ['sgd_conf']


# ── Stage 1: GRU model ────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=GRU_HIDDEN, num_layers=GRU_LAYERS, dropout=GRU_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return torch.sigmoid(self.fc(out))


class TradingLoss(nn.Module):
    """
    Combined loss with two components:

    1. Profit weight — wrong predictions on large price moves cost more,
       aligning training with actual P&L impact.

    2. Anti-lag penalty — technical indicators lag the market by design.
       When the model predicts the same direction as yesterday AND is wrong,
       the penalty scales up by lag_penalty. This pushes the model to
       anticipate reversals instead of blindly following lagging signals.
    """
    def __init__(self, lag_penalty: float = 1.5):
        super().__init__()
        self.lag_penalty = lag_penalty

    def forward(self, predictions, targets, price_changes, prev_directions):
        bce = F.binary_cross_entropy(predictions.squeeze(), targets, reduction='none')
        profit_weights = price_changes.abs() + 1.0
        pred_binary = (predictions.squeeze() >= 0.5).float()
        is_trend_following = (pred_binary == prev_directions).float()
        is_wrong = (pred_binary != targets).float()
        lag_weights = 1.0 + self.lag_penalty * is_trend_following * is_wrong
        return (bce * profit_weights * lag_weights).mean()


# ── Data pipeline (shared across all stages) ──────────────────────────────────

def fetch_historical_data(symbol, days=400):
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
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

        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + gain / loss))

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
        data['Price_Change_Pct'] = data['Close'].pct_change().shift(-1)

        data['ATR_TR'] = data.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Previous_Close']),
                abs(row['Low'] - row['Previous_Close'])
            ), axis=1
        )
        data['ATR_14'] = data['ATR_TR'].rolling(window=14).mean()

        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()


def add_sentiment(data, symbol):
    if not Finnhub_API_Key:
        data['sentiment'] = 0.0
        return data
    try:
        today = datetime.today()
        from_date = (today - timedelta(days=400)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
            f"&from={from_date}&to={to_date}&token={Finnhub_API_Key}"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200 or not response.json():
            data['sentiment'] = 0.0
            return data
        news_df = pd.DataFrame(response.json())
        if news_df.empty:
            data['sentiment'] = 0.0
            return data
        news_df['Timestamp'] = (
            pd.to_datetime(news_df['datetime'], unit='s')
            .dt.floor('D')
            .dt.tz_localize('UTC')
            .dt.tz_convert('America/New_York')
            .dt.normalize()
            .dt.tz_localize(None)
        )
        analyzer = SentimentIntensityAnalyzer()
        news_df['sentiment'] = news_df['headline'].apply(
            lambda t: analyzer.polarity_scores(t)['compound']
        )
        sentiment_by_date = news_df.groupby('Timestamp')['sentiment'].mean().reset_index()
        data = pd.merge(data, sentiment_by_date, how='left', on='Timestamp')
        data['sentiment'] = data['sentiment'].fillna(0.0)
        tm.sleep(1)
    except Exception as e:
        print(f"Sentiment error for {symbol}: {e}")
        data['sentiment'] = 0.0
    return data


def create_sequences(X, y, price_changes, prev_dirs, seq_length=SEQUENCE_LENGTH):
    Xs, ys, pcs, pds = [], [], [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
        pcs.append(price_changes[i + seq_length])
        pds.append(prev_dirs[i + seq_length])
    return np.array(Xs), np.array(ys), np.array(pcs), np.array(pds)


# ── Stage 1: Train / load GRU, get probability column ────────────────────────

def run_gru_stage(data, symbol, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_gru.pt")
    scaler_path = os.path.join(model_dir, f"{symbol}_gru_scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for col in BASE_FEATURES:
        if col not in data.columns:
            data[col] = 0.0

    X = data[BASE_FEATURES].values
    y = data['Direction'].values.astype(np.float32)
    price_changes = data['Price_Change_Pct'].values.astype(np.float32)
    prev_dirs = np.concatenate([[0.0], y[:-1]]).astype(np.float32)

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    X_seq, y_seq, pc_seq, pd_seq = create_sequences(X_scaled, y, price_changes, prev_dirs)

    n_test = TEST_SIZE
    X_train = X_seq[:-n_test]
    y_train = y_seq[:-n_test]
    pc_train = pc_seq[:-n_test]
    pd_train = pd_seq[:-n_test]

    model = GRUModel(input_size=len(BASE_FEATURES)).to(device)
    criterion = TradingLoss(lag_penalty=1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  [GRU] Loaded existing weights for {symbol}.")
    else:
        print(f"  [GRU] Training new model for {symbol}...")

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(pc_train, dtype=torch.float32),
            torch.tensor(pd_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    model.train()
    for epoch in range(GRU_EPOCHS):
        epoch_loss = 0.0
        for X_b, y_b, pc_b, pd_b in loader:
            X_b, y_b, pc_b, pd_b = X_b.to(device), y_b.to(device), pc_b.to(device), pd_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b, pc_b, pd_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(loader))

    torch.save(model.state_dict(), model_path)

    # Generate gru_prob for every row that has a full sequence preceding it
    model.eval()
    with torch.no_grad():
        all_seqs = torch.tensor(X_seq, dtype=torch.float32).to(device)
        all_probs = model(all_seqs).squeeze().cpu().numpy()

    # Align probabilities back to the original data index
    # X_seq[i] corresponds to data row SEQUENCE_LENGTH + i
    gru_prob_col = np.full(len(data), np.nan)
    for i, prob in enumerate(all_probs):
        gru_prob_col[SEQUENCE_LENGTH + i] = prob

    data = data.copy()
    data['gru_prob'] = gru_prob_col
    data_with_gru = data.dropna(subset=['gru_prob']).copy()

    gru_accuracy = accuracy_score(
        y_seq[-n_test:],
        (all_probs[-n_test:] >= 0.5).astype(int)
    )
    next_prob = float(all_probs[-1])
    print(f"  [GRU] Accuracy: {gru_accuracy:.2f} | Next prob: {next_prob:.3f}")

    return data_with_gru, gru_accuracy, next_prob


# ── Stage 2: SGD uses base features + gru_prob ───────────────────────────────

def run_sgd_stage(data, symbol, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_combined_sgd.pkl")

    train_data = data.iloc[:-TEST_SIZE]
    test_data = data.iloc[-TEST_SIZE:]

    X_train = train_data[EXTENDED_FEATURES].values
    y_train = train_data['Direction'].values
    X_test = test_data[EXTENDED_FEATURES].values
    y_test = test_data['Direction'].values
    close_prices_test = test_data['Close'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_cv_f1 = 0.0

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  [SGD] Loaded existing model for {symbol}.")
    else:
        model = SGDClassifier(random_state=42)
        param_grid = {
            'loss': ['hinge', 'log_loss'],
            'penalty': ['l1', 'l2'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['invscaling', 'adaptive'],
            'eta0': [0.01, 0.1, 0.001],
        }
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='f1', n_jobs=1)
        gs.fit(X_train_s, y_train)
        best_cv_f1 = gs.best_score_
        model = gs.best_estimator_
        print(f"  [SGD] Best CV F1: {best_cv_f1:.4f}")

    classes = np.unique(y_train)
    batch_size = 20
    for i in range(0, len(X_train_s), batch_size):
        model.partial_fit(X_train_s[i:i+batch_size], y_train[i:i+batch_size], classes=classes)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    predictions = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, predictions)

    # Decision confidence: distance from hyperplane (only for log_loss / hinge with coef_)
    try:
        raw_conf = model.decision_function(X_test_s)
        sgd_conf = 1 / (1 + np.exp(-raw_conf))  # sigmoid to [0,1]
    except Exception:
        sgd_conf = predictions.astype(float)

    last_10_accuracy = accuracy_score(y_test[-10:], predictions[-10:])
    print(f"  [SGD] Accuracy: {accuracy:.2f} | Last-10: {last_10_accuracy:.2f}")

    return {
        "predictions": predictions,
        "y_test": y_test,
        "accuracy": accuracy,
        "last_10_accuracy": last_10_accuracy,
        "best_cv_f1": best_cv_f1,
        "close_prices_test": close_prices_test,
        "sgd_conf": sgd_conf,
        "test_data": test_data,
        "scaler": scaler,
        "model": model,
    }


# ── Stage 3: RL uses base + gru_prob + sgd_conf ──────────────────────────────

class CombinedTradingEnv(gym.Env):
    """
    Trading environment whose observation vector includes:
      - base technical features
      - GRU probability (sequential signal)
      - SGD confidence (linear model signal)
    The RL agent learns when to trust each signal.
    """
    def __init__(self, data):
        super().__init__()
        self.action_space = spaces.Discrete(2)  # 0: Buy, 1: Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(RL_FEATURES),), dtype=np.float32
        )
        self.data = data.reset_index(drop=True)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_price = self.data['Close'].iloc[self.current_step]
        pct_change = (next_price - current_price) / current_price * 100
        reward = pct_change if action == 0 else -pct_change
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        return row[RL_FEATURES].values.astype(np.float32)


def run_rl_stage(data, sgd_result, symbol, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_combined_ppo.zip")

    # Attach sgd_conf to data for RL observation
    data = data.copy()
    # sgd_conf covers only the last TEST_SIZE rows; backfill with 0.5 for training rows
    full_sgd_conf = np.full(len(data), 0.5)
    full_sgd_conf[-TEST_SIZE:] = sgd_result["sgd_conf"]
    data['sgd_conf'] = full_sgd_conf

    # Ensure all RL features exist
    for col in RL_FEATURES:
        if col not in data.columns:
            data[col] = 0.0

    # Use only training portion for RL training, full data for evaluation
    train_data = data.iloc[:-TEST_SIZE].copy()
    test_data = data.iloc[-TEST_SIZE:].copy()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CombinedTradingEnv(train_data)
    vec_env = DummyVecEnv([lambda: env])

    if os.path.exists(model_path):
        rl_model = PPO.load(model_path, env=vec_env)
        print(f"  [RL] Loaded existing model for {symbol}.")
    else:
        rl_model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=0.001, seed=seed)
        print(f"  [RL] Training new model for {symbol}...")

    rl_model.learn(total_timesteps=8000)
    rl_model.save(model_path)

    # Evaluate on test data
    eval_env = CombinedTradingEnv(test_data)
    obs = eval_env.reset()
    done = False
    total_reward = 0.0
    win_count = 0
    total_trades = 0
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward
        total_trades += 1
        if reward > 0:
            win_count += 1

    win_rate = win_count / total_trades if total_trades > 0 else 0
    print(f"  [RL] Total reward: {float(total_reward):.2f} | Win rate: {win_rate:.2f}")

    return {
        "total_reward": float(total_reward),
        "win_rate": win_rate,
        "total_trades": total_trades,
    }


# ── Combined profit backtest (using SGD predictions on real prices) ───────────

def calculate_profit(predictions, close_prices, quantity):
    profit = 0.0
    for i in range(len(predictions) - 1):
        cur, nxt = close_prices[i], close_prices[i + 1]
        if predictions[i] == 1:
            profit += (nxt - cur) * quantity if nxt > cur else -(cur - nxt) * quantity
        else:
            profit += (cur - nxt) * quantity if nxt < cur else -(nxt - cur) * quantity
    return profit


# ── Main pipeline ─────────────────────────────────────────────────────────────

try:
    sp500_symbols = list(
        pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol']
    )
except Exception:
    sp500_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
    print("Could not fetch S&P 500 list, using default symbols.")

GRU_DIR = "Models_Combined_GRU"
SGD_DIR = "Models_Combined_SGD"
RL_DIR = "Models_Combined_RL"
ALLOCATION_PERCENT = 5


if __name__ == "__main__":
    all_results = []

    for symbol in sp500_symbols:
        try:
            print(f"\n{'='*60}")
            print(f"Pipeline: {symbol}")
            min_rows = TEST_SIZE + SEQUENCE_LENGTH + 30
            bars_df = fetch_historical_data(symbol)
            if bars_df.empty:
                continue

            data = add_features(bars_df)
            if data.empty or len(data) < min_rows:
                print(f"Not enough data for {symbol}, skipping.")
                continue

            data = add_sentiment(data, symbol)

            # ── Stage 1: GRU ──────────────────────────────────────────
            data_with_gru, gru_accuracy, gru_next_prob = run_gru_stage(data, symbol, GRU_DIR)
            if len(data_with_gru) < TEST_SIZE + 10:
                print(f"Not enough post-GRU rows for {symbol}, skipping.")
                continue

            # ── Stage 2: SGD ──────────────────────────────────────────
            sgd_result = run_sgd_stage(data_with_gru, symbol, SGD_DIR)

            # ── Stage 3: RL ───────────────────────────────────────────
            rl_result = run_rl_stage(data_with_gru, sgd_result, symbol, RL_DIR)

            # ── Backtest profit using SGD predictions ─────────────────
            current_price = sgd_result["close_prices_test"][-1]
            quantity = int((PORTFOLIO_SIZE * ALLOCATION_PERCENT / 100) // current_price)
            profit = calculate_profit(
                sgd_result["predictions"],
                sgd_result["close_prices_test"],
                quantity
            )

            sgd_acc = sgd_result["accuracy"]
            if sgd_acc <= 0.48:
                sgd_acc = 1 - sgd_acc

            result_row = {
                "Symbol": symbol,
                "GRU_Accuracy": round(gru_accuracy * 100, 2),
                "GRU_Next_Prob": round(gru_next_prob, 4),
                "GRU_Signal": "UP" if gru_next_prob >= 0.5 else "DOWN",
                "SGD_Accuracy": round(sgd_acc * 100, 2),
                "SGD_Accuracy_10": round(sgd_result["last_10_accuracy"] * 100, 2),
                "SGD_CV_F1": round(sgd_result["best_cv_f1"] * 100, 2),
                "RL_Win_Rate": round(rl_result["win_rate"] * 100, 2),
                "RL_Total_Reward": round(rl_result["total_reward"], 2),
                "Backtest_Profit": round(profit, 2),
                "Current_Price": round(current_price, 2),
            }
            all_results.append(result_row)
            print(f"  Result: {result_row}")

            # Save incrementally
            try:
                output_file = "Combined_results.xlsx"
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

    print("\nPipeline complete.")
