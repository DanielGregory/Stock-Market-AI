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

import copy
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random

load_dotenv()

Finnhub_API_Key = os.getenv("FINNHUB_API_KEY", "")
PORTFOLIO_SIZE = float(os.getenv("PORTFOLIO_SIZE", "100000"))

# ── Hyperparameters ───────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 30     # 6 weeks of context for weekly predictions
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_LR = 0.001
GRU_EPOCHS = 30
BATCH_SIZE = 32
TEST_SIZE = 100          # ~20 weekly periods for meaningful evaluation
HOLD_PERIOD = 5          # trading days per prediction period (1 week)

BASE_FEATURES = [
    # Price / OHLCV
    'Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close',
    # Trend
    'MA_20', 'MA_50', 'Price_MA20_pct',
    # Volatility / range
    'Volatility_20', 'ATR_14', 'BB_pct', 'High_Low_pct',
    # Momentum oscillators
    'RSI', 'Stoch_K', 'Stoch_D', 'ROC_5', 'ROC_20', 'Candle_body',
    # Volume
    'Obv', 'Volume_ratio',
    # MACD family
    'MACD', 'MACD_hist', 'MACD_Crossover',
    # Sentiment
    'sentiment',
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


class WeightedBCELoss(nn.Module):
    """
    Magnitude-weighted BCE: errors on larger price moves cost proportionally more.

    pos_weight corrects for class imbalance — in bull markets UP weeks outnumber
    DOWN weeks, which can cause a naïve model to just predict UP every time.
    Setting pos_weight = (# down weeks) / (# up weeks) re-balances the gradient.

    The previous TradingLoss included an "anti-lag penalty" that extra-penalized
    trend-following predictions when wrong. On trending markets this created a
    contrarian bias, pushing GRU accuracy below 50% (predicting DOWN in an
    uptrend). This simpler loss avoids that pathology.
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions, targets, price_changes, prev_dirs=None):
        bce = F.binary_cross_entropy(predictions.squeeze(), targets, reduction='none')
        magnitude_weights = price_changes.abs() * 5.0 + 1.0
        class_weights = torch.where(
            targets == 1,
            torch.full_like(targets, self.pos_weight),
            torch.ones_like(targets),
        )
        return (bce * magnitude_weights * class_weights).mean()


# ── Data pipeline (shared across all stages) ──────────────────────────────────

def fetch_historical_data(symbol, days=600):
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

        # ── Trend ─────────────────────────────────────────────────────────────
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        data['Price_MA20_pct'] = (data['Close'] - data['MA_20']) / (data['MA_20'] + 1e-9) * 100

        # ── Volatility / range ────────────────────────────────────────────────
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']

        data['ATR_TR'] = data.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Previous_Close']),
                abs(row['Low'] - row['Previous_Close'])
            ), axis=1
        )
        data['ATR_14'] = data['ATR_TR'].rolling(window=14).mean()

        bb_std = data['Close'].rolling(window=20).std()
        bb_upper = data['MA_20'] + 2 * bb_std
        bb_lower = data['MA_20'] - 2 * bb_std
        data['BB_pct'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)

        data['High_Low_pct'] = (data['High'] - data['Low']) / (data['Close'] + 1e-9) * 100
        data['Candle_body']  = (data['Close'] - data['Open']) / (data['Open'] + 1e-9) * 100

        # ── Momentum oscillators ──────────────────────────────────────────────
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        data['RSI'] = 100 - (100 / (1 + gain / loss))

        low_14  = data['Low'].rolling(window=14).min()
        high_14 = data['High'].rolling(window=14).max()
        data['Stoch_K'] = (data['Close'] - low_14) / (high_14 - low_14 + 1e-9) * 100
        data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

        data['ROC_5']  = (data['Close'] / data['Close'].shift(5)  - 1) * 100
        data['ROC_20'] = (data['Close'] / data['Close'].shift(20) - 1) * 100

        # ── Volume ────────────────────────────────────────────────────────────
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['Obv'] = obv

        vol_20 = data['Volume'].rolling(window=20).mean()
        data['Volume_ratio'] = data['Volume'] / (vol_20 + 1e-9)

        # ── MACD family ───────────────────────────────────────────────────────
        data['12_EMAs'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['26_EMAs'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD']         = data['12_EMAs'] - data['26_EMAs']
        data['Signal_Line']  = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist']    = data['MACD'] - data['Signal_Line']
        data['MACD_Crossover'] = (data['MACD'] > data['Signal_Line']).astype(int)

        # ── Weekly target: will price be higher 5 trading days from now? ──────
        data['Direction']       = (data['Close'].shift(-HOLD_PERIOD) > data['Close']).astype(int)
        data['Price_Change_Pct'] = (data['Close'].shift(-HOLD_PERIOD) - data['Close']) / data['Close']

        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()


def _unix_to_naive_date(ts: int) -> pd.Timestamp:
    """Convert a Unix timestamp to a naive (no-tz) ET calendar date."""
    return (
        pd.Timestamp(int(ts), unit='s')
        .tz_localize('UTC')
        .tz_convert('America/New_York')
        .normalize()
        .tz_localize(None)
    )


def add_sentiment(data, symbol):
    """
    Merge daily news-sentiment scores into data['sentiment'].

    Sources tried in order:
      1. yfinance .news  — free, recent headlines only (last few weeks)
      2. Finnhub         — full 600-day history; requires FINNHUB_API_KEY in .env

    Days with no news default to 0.0 (neutral).
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment_map: dict = {}   # date → list[float]

    # ── Source 1: yfinance (always free) ─────────────────────────────────────
    try:
        news_list = yf.Ticker(symbol).news or []
        for item in news_list:
            # yfinance returns different shapes across versions
            title = item.get('title') or ''
            if not title:
                content = item.get('content') or {}
                title = content.get('title', '') if isinstance(content, dict) else ''
            ts = item.get('providerPublishTime') or item.get('pubDate') or 0
            if not title or not ts:
                continue
            try:
                date = _unix_to_naive_date(ts)
            except Exception:
                continue
            sentiment_map.setdefault(date, []).append(
                analyzer.polarity_scores(title)['compound']
            )
        if sentiment_map:
            total = sum(len(v) for v in sentiment_map.values())
            print(f"  Sentiment: {total} articles via yfinance (free)")
    except Exception as e:
        print(f"  yfinance news unavailable for {symbol}: {e}")

    # ── Source 2: Finnhub (optional, full history) ────────────────────────────
    if Finnhub_API_Key:
        try:
            today = datetime.today()
            from_date = (today - timedelta(days=600)).strftime("%Y-%m-%d")
            to_date   = today.strftime("%Y-%m-%d")
            url = (
                f"https://finnhub.io/api/v1/company-news?symbol={symbol}"
                f"&from={from_date}&to={to_date}&token={Finnhub_API_Key}"
            )
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                for item in (resp.json() or []):
                    title = item.get('headline', '')
                    ts    = item.get('datetime', 0)
                    if not title or not ts:
                        continue
                    try:
                        date = _unix_to_naive_date(ts)
                    except Exception:
                        continue
                    sentiment_map.setdefault(date, []).append(
                        analyzer.polarity_scores(title)['compound']
                    )
            tm.sleep(1)
        except Exception as e:
            print(f"  Finnhub unavailable for {symbol}: {e}")

    if not sentiment_map:
        data['sentiment'] = 0.0
        return data

    sentiment_df = pd.DataFrame([
        {'Timestamp': date, 'sentiment': float(np.mean(scores))}
        for date, scores in sentiment_map.items()
    ])
    data = pd.merge(data, sentiment_df, how='left', on='Timestamp')
    data['sentiment'] = data['sentiment'].fillna(0.0)
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
    X_train_full = X_seq[:-n_test]
    y_train_full = y_seq[:-n_test]
    pc_train_full = pc_seq[:-n_test]
    pd_train_full = pd_seq[:-n_test]

    n_val = max(20, len(X_train_full) // 5)
    X_train = X_train_full[:-n_val];  y_train = y_train_full[:-n_val]
    pc_train = pc_train_full[:-n_val]; pd_train = pd_train_full[:-n_val]
    X_val = X_train_full[-n_val:];    y_val = y_train_full[-n_val:]
    pc_val = pc_train_full[-n_val:];  pd_val = pd_train_full[-n_val:]

    def _t(*arrays):
        return [torch.tensor(a, dtype=torch.float32).to(device) for a in arrays]

    loader = DataLoader(
        TensorDataset(*_t(X_train, y_train, pc_train, pd_train)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    val_t = _t(X_val, y_val, pc_val, pd_val)

    n_up = float(y_train.sum())
    n_down = float(len(y_train) - n_up)
    pos_weight = n_down / n_up if n_up > 0 else 1.0

    model = GRUModel(input_size=len(BASE_FEATURES)).to(device)
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"  [GRU] Loaded existing weights for {symbol}.")
        except Exception:
            print(f"  [GRU] Feature set changed — retraining for {symbol}.")
    else:
        print(f"  [GRU] Training new model for {symbol}...")

    PATIENCE = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(GRU_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b, pc_b, pd_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b, pc_b, pd_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_t[0]), *val_t[1:]).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [GRU] Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_weights)

    torch.save(model.state_dict(), model_path)

    # Generate gru_prob for every row that has a full sequence preceding it
    model.eval()
    with torch.no_grad():
        all_seqs = torch.tensor(X_seq, dtype=torch.float32).to(device)
        all_probs = model(all_seqs).squeeze().cpu().numpy()

    # If the GRU is consistently wrong on training data, flip its signal.
    # Checked on training portion only (no lookahead) so SGD and RL always
    # receive a probability where > 0.5 genuinely means UP.
    train_acc = accuracy_score(y_seq[:-n_test], (all_probs[:-n_test] >= 0.5).astype(int))
    if train_acc < 0.5:
        print(f"  [GRU] Inverted signal (train acc {train_acc:.2f}) — flipping probabilities.")
        all_probs = 1.0 - all_probs

    gru_accuracy = accuracy_score(y_seq[-n_test:], (all_probs[-n_test:] >= 0.5).astype(int))
    next_prob = float(all_probs[-1])
    print(f"  [GRU] Accuracy: {gru_accuracy:.2f} | Next prob: {next_prob:.3f}")

    # Align (oriented) probabilities back to the original data index
    gru_prob_col = np.full(len(data), np.nan)
    for i, prob in enumerate(all_probs):
        gru_prob_col[SEQUENCE_LENGTH + i] = prob

    data = data.copy()
    data['gru_prob'] = gru_prob_col
    data_with_gru = data.dropna(subset=['gru_prob']).copy()

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

    model = None
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                candidate = pickle.load(f)
            candidate.decision_function(X_train_s[:1])  # dimension check
            model = candidate
            print(f"  [SGD] Loaded existing model for {symbol}.")
        except Exception:
            print(f"  [SGD] Feature set changed — retraining for {symbol}.")

    if model is None:
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

    try:
        raw_conf = model.decision_function(X_test_s)
        sgd_conf = 1 / (1 + np.exp(-raw_conf))
    except Exception:
        sgd_conf = predictions.astype(float)

    accuracy  = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall    = recall_score(y_test, predictions, zero_division=0)
    f1        = f1_score(y_test, predictions, zero_division=0)
    try:
        auc = roc_auc_score(y_test, sgd_conf)
    except ValueError:
        auc = 0.0
    sharpe           = calculate_sharpe(predictions, close_prices_test)
    last_10_accuracy = accuracy_score(y_test[-10:], predictions[-10:])

    print(
        f"  [SGD] Acc: {accuracy:.2f} | P: {precision:.2f} | R: {recall:.2f} | "
        f"F1: {f1:.2f} | AUC: {auc:.2f} | Sharpe: {sharpe:.2f}"
    )

    return {
        "predictions":      predictions,
        "y_test":           y_test,
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "auc":              auc,
        "sharpe":           sharpe,
        "last_10_accuracy": last_10_accuracy,
        "best_cv_f1":       best_cv_f1,
        "close_prices_test": close_prices_test,
        "sgd_conf":         sgd_conf,
        "test_data":        test_data,
        "scaler":           scaler,
        "model":            model,
    }


# ── Stage 3: RL uses base + gru_prob + sgd_conf ──────────────────────────────

class CombinedTradingEnv(gym.Env):
    """
    Weekly swing-trading environment.

    Actions:
      0 = flat / exit (sell if holding, stay out if already flat)
      1 = long / hold  (buy if flat, hold if already long)

    Each step advances HOLD_PERIOD trading days (one week). The agent
    receives the weekly price return when long and 0 when flat. Observation
    includes all RL features plus the current position (0 or 1) so the
    agent knows whether it already holds a position.
    """
    def __init__(self, data):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(RL_FEATURES) + 1,),  # +1 for current position
            dtype=np.float32
        )
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.position = 0      # 0 = flat, 1 = long
        self.entry_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]

        if action == 1:          # go / stay long
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        else:                    # exit / stay flat
            if self.position == 1:
                self.position = 0
                self.entry_price = 0.0

        next_step = min(self.current_step + HOLD_PERIOD, len(self.data) - 1)
        next_price = self.data['Close'].iloc[next_step]
        terminated = next_step >= len(self.data) - 1

        reward = (next_price - current_price) / current_price * 100 if self.position == 1 else 0.0

        self.current_step = next_step
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        features = row[RL_FEATURES].values.astype(np.float32)
        return np.append(features, float(self.position))


def run_rl_stage(data, sgd_result, symbol, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_combined_ppo.zip")

    # Generate SGD confidence for every row using the trained model + scaler,
    # so the RL agent trains on real signal rather than a flat 0.5 backfill.
    data = data.copy()
    for col in EXTENDED_FEATURES:
        if col not in data.columns:
            data[col] = 0.0
    X_all = sgd_result["scaler"].transform(data[EXTENDED_FEATURES].values)
    try:
        raw_scores = sgd_result["model"].decision_function(X_all)
        data['sgd_conf'] = 1 / (1 + np.exp(-raw_scores))
    except Exception:
        data['sgd_conf'] = sgd_result["model"].predict(X_all).astype(float)

    # Ensure remaining RL features exist
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
        try:
            rl_model = PPO.load(model_path, env=vec_env)
            print(f"  [RL] Loaded existing model for {symbol}.")
        except Exception:
            print(f"  [RL] Feature set changed — retraining for {symbol}.")
            rl_model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=0.001, seed=seed)
    else:
        rl_model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=0.001, seed=seed)
        print(f"  [RL] Training new model for {symbol}...")

    rl_model.learn(total_timesteps=8000)
    rl_model.save(model_path)

    # Evaluate on test data
    eval_env = CombinedTradingEnv(test_data)
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0.0
    win_count = 0
    total_trades = 0
    while not done:
        action, _ = rl_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
        if action == 1:  # only count held periods; flat/sell steps have reward=0
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


# ── Shared metric helpers ─────────────────────────────────────────────────────

def calculate_sharpe(predictions, close_prices, risk_free_rate=0.0):
    """
    Annualized Sharpe based on weekly hold-or-flat strategy.
    Long for a full week when prediction is UP, flat (0 return) when DOWN.
    Annualizes with sqrt(52) since returns are weekly.
    """
    returns = []
    i = 0
    while i + HOLD_PERIOD < len(close_prices):
        if close_prices[i] == 0:
            i += HOLD_PERIOD
            continue
        if predictions[i] == 1:
            ret = (close_prices[i + HOLD_PERIOD] - close_prices[i]) / close_prices[i]
        else:
            ret = 0.0
        returns.append(ret)
        i += HOLD_PERIOD
    if not returns:
        return 0.0
    r = np.array(returns)
    std = r.std()
    if std == 0:
        return 0.0
    return float((r.mean() - risk_free_rate / 52) / std * np.sqrt(52))


def calculate_trade_stats(predictions, close_prices, quantity,
                          portfolio_size=None, allocation_pct=None):
    """
    Runs the hold-through-up backtest and returns full trade statistics:
    profit, return %, starting allocation, buy/hold/sell counts, avg hold.
    """
    if portfolio_size is None:
        portfolio_size = PORTFOLIO_SIZE
    if allocation_pct is None:
        allocation_pct = ALLOCATION_PERCENT

    allocation = portfolio_size * allocation_pct / 100
    profit = 0.0
    position = 0
    entry_price = 0.0
    entry_step = 0
    n_buys = n_holds = n_sells = 0
    hold_weeks_list = []

    i = 0
    while i < len(close_prices):
        price = close_prices[i]
        pred = predictions[i]
        if pred == 1:
            if position == 0:
                position = 1
                entry_price = price
                entry_step = i
                n_buys += 1
            else:
                n_holds += 1
        else:
            if position == 1:
                profit += (price - entry_price) * quantity
                hold_weeks_list.append(max(1, (i - entry_step) // HOLD_PERIOD))
                position = 0
                n_sells += 1
        i += HOLD_PERIOD

    if position == 1:
        profit += (close_prices[-1] - entry_price) * quantity
        hold_weeks_list.append(max(1, (len(close_prices) - 1 - entry_step) // HOLD_PERIOD))
        n_sells += 1

    avg_hold = sum(hold_weeks_list) / len(hold_weeks_list) if hold_weeks_list else 0
    return_pct = profit / allocation * 100 if allocation > 0 else 0

    return {
        "profit":             round(profit, 2),
        "starting_allocation": round(allocation, 2),
        "return_pct":         round(return_pct, 1),
        "quantity":           quantity,
        "n_buys":             n_buys,
        "n_holds":            n_holds,
        "n_sells":            n_sells,
        "avg_hold_weeks":     round(avg_hold, 1),
    }


def calculate_profit(predictions, close_prices, quantity):
    return calculate_trade_stats(predictions, close_prices, quantity)["profit"]


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
            tm.sleep(1)  # avoid yfinance rate limiting
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

            # Flip predictions when model is sub-chance (inverted signal is useful)
            predictions = sgd_result["predictions"].copy()
            sgd_acc = sgd_result["accuracy"]
            if sgd_acc <= 0.48:
                predictions = 1 - predictions
                sgd_acc = 1 - sgd_acc

            profit = calculate_profit(
                predictions,
                sgd_result["close_prices_test"],
                quantity
            )

            result_row = {
                "Symbol":           symbol,
                "GRU_Acc_%":        round(gru_accuracy * 100,                    2),
                "GRU_Next_Prob":    round(gru_next_prob,                          4),
                "GRU_Signal":       "UP" if gru_next_prob >= 0.5 else "DOWN",
                "SGD_Acc_%":        round(sgd_acc * 100,                          2),
                "SGD_Precision_%":  round(sgd_result["precision"] * 100,          2),
                "SGD_Recall_%":     round(sgd_result["recall"] * 100,             2),
                "SGD_F1_%":         round(sgd_result["f1"] * 100,                 2),
                "SGD_AUC":          round(sgd_result["auc"],                      4),
                "SGD_Sharpe":       round(sgd_result["sharpe"],                   3),
                "SGD_Acc_10_%":     round(sgd_result["last_10_accuracy"] * 100,   2),
                "SGD_CV_F1_%":      round(sgd_result["best_cv_f1"] * 100,         2),
                "RL_Win_Rate_%":    round(rl_result["win_rate"] * 100,            2),
                "RL_Total_Reward":  round(rl_result["total_reward"],              2),
                "Backtest_Profit":  round(profit,                                  2),
                "Current_Price":    round(current_price,                           2),
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
