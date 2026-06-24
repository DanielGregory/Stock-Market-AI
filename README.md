# Stock Market AI

A machine learning pipeline for daily stock direction prediction. Three models — a GRU recurrent network, an SGD Classifier, and a PPO Reinforcement Learning agent — are chained in sequence so each stage feeds signal into the next.

> **Disclaimer:** This is a research and learning project. Nothing here constitutes financial advice. Past model performance on historical data does not predict future returns.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [The Three Models](#the-three-models)
- [Feature Engineering](#feature-engineering)
- [Custom Loss Function](#custom-loss-function-tradingloss)
- [Key Design Decisions](#key-design-decisions)
- [Quick Start](#quick-start)
- [Running the Demo](#running-the-demo)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Limitations](#limitations)

---

## Architecture Overview

The core insight driving the design: each model type captures a different kind of signal, and they have different blind spots. Chaining them sequentially means each stage gets access to what the previous stage learned.

```
Raw OHLCV Data (Yahoo Finance)
          │
          ▼
  Feature Engineering
  RSI · MACD · OBV · ATR · MA · Volume · Sentiment
          │
          ├─────────────────────────────────────┐
          │                                     │
          ▼                                     │
  ┌───────────────┐                             │
  │  Stage 1      │                             │
  │  GRU Network  │──► gru_prob                 │
  │  20-day seqs  │    (sequence confidence)    │
  └───────────────┘                             │
          │                                     │
          ▼                                     │
  ┌───────────────┐◄────────────────────────────┘
  │  Stage 2      │  base features + gru_prob
  │  SGD          │──► prediction + sgd_conf
  │  Classifier   │    (linear confidence score)
  └───────────────┘
          │
          ▼
  ┌───────────────┐
  │  Stage 3      │  base features + gru_prob + sgd_conf
  │  PPO RL Agent │──► Buy / Sell
  │               │    (learns when to trust each signal)
  └───────────────┘
```

The RL agent sees all three signals simultaneously and learns, through reward, when to act on them and when to ignore them.

---

## The Three Models

### Stage 1 — GRU (`GRU_Model.py`)

A Gated Recurrent Unit network that processes 20-day sequences of technical features. Unlike the SGD which sees each day in isolation, the GRU can learn temporal patterns — things like "three consecutive days of declining volume followed by an RSI dip often precede a bounce."

**Architecture:**
- 2-layer GRU, hidden size 64, dropout 0.2
- Input: 20 × 13 feature matrix (20 trading days, 13 features each)
- Output: single probability (0–1) of next-day upward movement
- Trained with `TradingLoss` (see below)
- Saves per-stock `.pt` weight files; supports incremental retraining

**Why GRU over LSTM?** GRUs have fewer parameters and converge faster on shorter sequences. With only 20-day windows and limited data per stock, LSTM's extra complexity doesn't pay off. Transformers were considered but the sequence length is too short to benefit from self-attention.

---

### Stage 2 — SGD Classifier (`AI_Test.py`)

A linear classifier trained with stochastic gradient descent, extended to accept `gru_prob` from Stage 1 as an additional feature. Where the GRU captures temporal context, the SGD draws a linear decision boundary across all features simultaneously.

**Key choices:**
- **Online learning (`partial_fit`):** The model updates in mini-batches rather than refitting from scratch each run. This lets it adapt to new data without forgetting old patterns.
- **Grid search with `TimeSeriesSplit`:** Hyperparameter tuning uses time-series cross-validation (not random splits) to avoid leakage — the validation set is always chronologically after the training set.
- **Flip logic:** If overall accuracy falls at or below 48%, predictions are inverted. A model that's consistently wrong more than 50% of the time is actually informative — you just need to bet the other way.

---

### Stage 3 — PPO Reinforcement Learning (`RL_Model.py`)

A Proximal Policy Optimization agent that learns a trading strategy through reward. Its observation space is the widest of the three: it sees all base features plus `gru_prob` and `sgd_conf` (SGD's sigmoid-transformed decision confidence).

**Why RL as the final layer?** The SGD outputs a direction prediction, but not a strategy. The RL agent learns *when* acting on that signal is profitable. If the GRU and SGD agree, the agent learns that's a higher-confidence signal. If they disagree, it learns to be cautious. This emergent risk management is what RL adds over simple ensembling.

**Reward function:** Percentage price change per step, positive for correct-direction trades and negative for incorrect ones. This keeps rewards comparable across stocks with very different price levels.

---

## Feature Engineering

Every feature is computed from OHLCV data fetched via `yfinance`. No external data sources are required to run the models (Finnhub news sentiment is optional).

| Feature | What it captures | Why it's here |
|---|---|---|
| `MA_20` | 20-day moving average of close | Medium-term trend direction |
| `Volatility_20` | 20-day rolling std of close | Risk level / regime detection |
| `RSI` | 14-period Relative Strength Index | Overbought / oversold momentum |
| `MACD_Crossover` | Binary: MACD above signal line | Momentum regime change signal |
| `OBV` | On-Balance Volume (cumulative) | Volume confirms or contradicts price moves |
| `ATR_14` | 14-day Average True Range | True volatility including gaps |
| `Volume_5` | 5-day average volume | Unusual activity detection |
| `Previous_Close` | Prior day's closing price | Anchors next-day change calculation |
| `Momentum` | Close minus Previous_Close | Raw one-day price velocity |
| `sentiment` | VADER score on Finnhub headlines | News-driven price pressure |

**Target variable:** `Direction` — 1 if tomorrow's close is above today's close, 0 otherwise. The GRU and SGD both predict this binary outcome.

---

## Custom Loss Function: `TradingLoss`

Standard binary cross-entropy treats every wrong prediction equally. That's fine for pure classification but poorly aligned with trading, where two specific failure modes matter more than others.

`TradingLoss` combines two penalty terms:

### 1. Profit weight

A wrong prediction on a 5% price move costs the same as a wrong prediction on a 0.1% move under plain BCE. In trading, those errors have vastly different P&L consequences. The profit weight scales each sample's loss by the magnitude of the next-day price change:

```
profit_weight = |price_change_pct| + 1.0
```

The `+1.0` ensures zero-change days still contribute to training.

### 2. Anti-lag penalty

Technical indicators are inherently backward-looking. MA_20 is an average of the last 20 closes. RSI looks back 14 periods. MACD compares two historical EMAs. A model trained on these features naturally learns to follow trends — it predicts "up" when all the lagging indicators say "up," which means it's always a step behind trend reversals.

The anti-lag penalty fires when the model predicts the *same direction as yesterday* AND is wrong — the classic lag pattern:

```python
is_trend_following = (predicted_direction == yesterday_direction)
is_wrong = (predicted_direction != actual_direction)
lag_weight = 1.0 + lag_penalty * is_trend_following * is_wrong
```

With `lag_penalty=1.5`, trend-following errors cost 2.5× more than contrarian errors. Over training, this pushes the GRU to pay attention to sequential patterns that precede reversals rather than just amplifying lagging signals.

**Combined loss:**
```
loss = mean(BCE * profit_weight * lag_weight)
```

---

## Key Design Decisions

**Sequential pipeline over ensemble voting**

A simple ensemble would average predictions from all three models. The sequential pipeline is more powerful because each stage sees what the previous stage learned. The RL agent doesn't just average signals — it learns *when* each signal is trustworthy based on the history of their combined performance.

**`TimeSeriesSplit` for cross-validation**

Using random k-fold CV on time series data leaks the future into the past — validation samples can occur before training samples chronologically. `TimeSeriesSplit` ensures the validation window is always after the training window, giving an honest estimate of how the model performs on unseen future data.

**Incremental training on the SGD**

`SGDClassifier.partial_fit()` updates the model on new data without starting over. When you run the pipeline again next week, it adapts to the new bars rather than discarding everything it learned previously. This is how the model stays relevant over time without expensive full retraining.

**Per-stock model files**

Each stock gets its own model weights (`{SYMBOL}_gru.pt`, `{SYMBOL}_sgd_model.pkl`, `{SYMBOL}_ppo_model.zip`). Stock behaviour varies enough — volatility regimes, sector dynamics, market cap effects — that a single shared model would average out meaningful per-stock patterns.

**Prediction flipping in the SGD**

If a model's test accuracy is ≤ 48%, it's performing worse than a coin flip. But a model that's consistently *wrong* is still useful — just invert its predictions. The flip logic converts a reliably wrong model into a reliably right one without discarding the learned weights.

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# 1. Clone and enter the repo
git clone https://github.com/DanielGregory/Stock-Market-AI.git
cd Stock-Market-AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up API keys
cp .env.example .env
# Edit .env and add your FINNHUB_API_KEY if you have one.
# The models run fine without it — sentiment defaults to 0.
```

---

## Running the Demo

The demo runs the full three-stage pipeline on 5 stocks with reduced training time (~2–5 minutes per stock on CPU).

```bash
# Default: AAPL, MSFT, GOOGL, TSLA, NVDA
python demo.py

# Custom symbols
python demo.py --symbols AAPL JPM AMZN

# Faster (less accurate — useful for a quick sanity check)
python demo.py --epochs 5 --steps 500
```

**Expected output:**

```
============================================================
  STOCK MARKET AI — DEMO
============================================================
  Symbols : AAPL MSFT GOOGL TSLA NVDA
  GRU epochs    : 10
  RL timesteps  : 2000
  Sentiment     : disabled

------------------------------------------------------------
  Processing: AAPL
------------------------------------------------------------
  Fetching AAPL data from Yahoo Finance...
  Data ready: 348 rows

  ─────────────────────────────────────────────────────
  Stage 1 / 3 — GRU (temporal pattern recognition)
  ...
  [GRU] Accuracy: 0.54 | Next prob: 0.67

  Stage 2 / 3 — SGD Classifier
  ...
  [SGD] Accuracy: 0.56 | Last-10: 0.60

  Stage 3 / 3 — PPO Reinforcement Learning
  ...
  [RL] Total reward: 12.34 | Win rate: 0.58

============================================================
  DEMO RESULTS SUMMARY
============================================================
  Symbol  GRU Acc %  GRU Signal  SGD Acc %  RL Win Rate %  Backtest $
  ...
```

Demo model weights are saved in `Demo_Models/` and won't overwrite any trained full-run models.

---

## Running the Full Pipeline

Each script can be run independently or together.

```bash
# Standalone SGD model on all S&P 500 stocks
python AI_Test.py

# Standalone RL model on all S&P 500 stocks
python RL_Model.py

# Standalone GRU model on all S&P 500 stocks
python GRU_Model.py

# Full sequential pipeline (GRU → SGD → RL) on all S&P 500 stocks
python Combined_Model.py
```

All scripts:
- Fetch the current S&P 500 list from Wikipedia on startup (falls back to 10 default stocks if unreachable)
- Save model weights after each stock so progress is preserved if interrupted
- Save/update results to an Excel file incrementally

---

## Output Files

| File | Contents |
|---|---|
| `SGD_results.xlsx` | Per-stock accuracy, CV score, predicted profit, top feature weight |
| `RL_results.xlsx` | Per-stock total reward, win rate, trade count |
| `GRU_results.xlsx` | Per-stock accuracy, next-day direction and probability |
| `Combined_results.xlsx` | All of the above merged, plus backtested profit |
| `demo_results.xlsx` | Results from the last `demo.py` run |
| `Models_SGD/{SYMBOL}/` | Saved SGD `.pkl` files |
| `Models_RL/{SYMBOL}/` | Saved PPO `.zip` files |
| `Models_GRU/{SYMBOL}/` | Saved GRU `.pt` and scaler `.pkl` files |
| `Models_Combined_*/` | Saved weights for the combined pipeline run |
| `Demo_Models/` | Weights from demo runs (separate from trained models) |

---

## Project Structure

```
Stock-Market-AI/
├── AI_Test.py          # Standalone SGD classifier
├── RL_Model.py         # Standalone PPO RL agent
├── GRU_Model.py        # Standalone GRU network
├── Combined_Model.py   # Sequential pipeline (GRU → SGD → RL)
├── demo.py             # Quick demo runner (5 stocks, reduced training)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .env                # Your local keys (never committed)
```

---

## Limitations

**Backtesting is not live performance.** The profit figures in the output files are computed on the held-out test set (last 50 trading days). This is a reasonable evaluation but not a guarantee of future results. There is no modelling of transaction costs, slippage, or market impact.

**Small test sets.** 50 days is a narrow evaluation window. Accuracy figures can swing significantly with market regime changes (high-volatility periods, macro events).

**Lagging indicators.** Despite the anti-lag penalty in `TradingLoss`, all features are derived from past prices. The models cannot anticipate events that have no historical precedent.

**The flip logic is a heuristic.** Inverting predictions when accuracy ≤ 48% assumes the model's errors are systematic rather than random. This holds sometimes but not always.

**The RL agent is simple.** The action space is binary (Buy / Sell) with no Hold option and no position sizing. A real trading strategy would need richer actions and explicit risk constraints.
