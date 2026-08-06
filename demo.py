"""
Demo runner — runs the full sequential pipeline on a small set of stocks
with reduced training time so you can see results in minutes.

Usage:
    python demo.py                            # runs default 5 stocks
    python demo.py --symbols AAPL MSFT TSLA  # custom list
    python demo.py --epochs 5 --steps 500    # even faster (lower accuracy)
"""

import argparse
import json
import os
import sys
import time
import datetime
import pandas as pd
import yfinance as yf

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Stock Market AI — Demo")
parser.add_argument(
    "--symbols", nargs="+",
    default=[
        # Tech — mega-cap
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL",
        # Tech — growth / mid
        "AMD", "QCOM", "CRM", "NFLX", "INTC",
        # Semis / hardware
        "MU", "GLW",
        # Cybersecurity / fintech / speculative
        "CRWD", "PLTR", "COIN", "NBIS",
        # Financials
        "JPM", "GS", "V", "MA", "BAC", "MS",
        # Healthcare
        "LLY", "UNH", "JNJ", "ABBV", "MRK",
        # Consumer discretionary
        "TSLA", "HD", "MCD", "NKE", "COST",
        # Consumer staples
        "WMT", "PG", "KO",
        # Energy
        "XOM", "CVX",
        # Industrials
        "CAT", "HON", "GE", "BA",
        # Momentum ETF
        "SPMO",
    ],
    help="Stock ticker symbols to run"
)
parser.add_argument(
    "--epochs", type=int, default=10,
    help="GRU training epochs (default: 10, full run uses 30)"
)
parser.add_argument(
    "--steps", type=int, default=2000,
    help="RL training timesteps (default: 2000, full run uses 8000)"
)
parser.add_argument(
    "--mode", choices=["train", "infer"], default="train",
    help="'train' does full retrain; 'infer' adds one data point to saved models (~3-5 min)"
)
parser.add_argument(
    "--profile", choices=["conservative", "aggressive"], default="conservative",
    help="'conservative' = default balanced model; 'aggressive' = concentrated high-volatility test sandbox"
)
args = parser.parse_args()

# ── Patch hyperparameters before importing the pipeline ───────────────────────
# This overrides the module-level constants in Combined_Model so the demo
# runs faster without changing the actual training files.

import Combined_Model as pipeline

pipeline.GRU_EPOCHS = args.epochs
pipeline.BATCH_SIZE = 16  # smaller batches for small datasets

DEMO_RL_STEPS = args.steps

# ── Profile configuration ─────────────────────────────────────────────────────
# Aggressive profile: concentrated, high-volatility, shorter hold, heavier
# expected-move weighting. Completely isolated from the conservative run.
if args.profile == "aggressive":
    DEMO_GRU_DIR    = "Aggressive_Models/GRU"
    DEMO_SGD_DIR    = "Aggressive_Models/SGD"
    DEMO_RL_DIR     = "Aggressive_Models/RL"
    RESULTS_PATH    = os.path.join("web", "data", "results_aggressive.json")
    PROFILE_MAX_POS = 5       # concentrated — fewer, bigger bets
    PROFILE_CAP_PCT = 40.0    # higher per-position cap
    pipeline.HOLD_PERIOD = 3  # 3-day hold → more active
else:
    DEMO_GRU_DIR    = "Demo_Models/GRU"
    DEMO_SGD_DIR    = "Demo_Models/SGD"
    DEMO_RL_DIR     = "Demo_Models/RL"
    RESULTS_PATH    = os.path.join("web", "data", "results.json")
    PROFILE_MAX_POS = 10
    PROFILE_CAP_PCT = 25.0

# In infer mode, carry over backtest metrics from the last full-train run so
# the dashboard doesn't lose historical accuracy figures.
_prev_results_map = {}
if args.mode == "infer":
    _prev_json = RESULTS_PATH
    if os.path.exists(_prev_json):
        try:
            with open(_prev_json) as _pf:
                _prev_data = json.load(_pf)
            for _r in _prev_data.get("results", []):
                _prev_results_map[_r["symbol"]] = _r
            print(f"  Loaded {len(_prev_results_map)} previous results for carry-over.")
        except Exception as _e:
            print(f"  Warning: could not load previous results: {_e}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(text, width=60, char="="):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

def section(text):
    print(f"\n  {'─' * 50}")
    print(f"  {text}")
    print(f"  {'─' * 50}")


def _try_infer(symbol, prev_map):
    """
    Fast inference path — no retraining.

    Loads the six saved model artifacts for *symbol*, fetches ~120 days of
    data, runs a GRU forward pass + SGD partial_fit on the newly resolved
    5-day label + a single RL observation, and returns a result row.

    Falls back to None (triggering a full train) if any artifact is missing
    or any step raises an exception.
    """
    import pickle as _pickle
    import numpy as _np
    import torch as _torch
    import pandas as _pd

    gru_model_path  = os.path.join(DEMO_GRU_DIR, f"{symbol}_gru.pt")
    gru_scaler_path = os.path.join(DEMO_GRU_DIR, f"{symbol}_gru_scaler.pkl")
    gru_meta_path   = os.path.join(DEMO_GRU_DIR, f"{symbol}_gru_meta.json")
    sgd_model_path  = os.path.join(DEMO_SGD_DIR, f"{symbol}_combined_sgd.pkl")
    sgd_scaler_path = os.path.join(DEMO_SGD_DIR, f"{symbol}_sgd_scaler.pkl")
    rl_model_path   = os.path.join(DEMO_RL_DIR,  f"{symbol}_combined_ppo.zip")

    for path in [gru_model_path, gru_scaler_path, gru_meta_path,
                 sgd_model_path, sgd_scaler_path, rl_model_path]:
        if not os.path.exists(path):
            print(f"  [INFER] {symbol}: missing {os.path.basename(path)}, will train.")
            return None

    try:
        # ── Fetch & engineer features ──────────────────────────────────────
        bars_df = pipeline.fetch_historical_data(symbol, days=120)
        time.sleep(0.5)
        if bars_df.empty:
            return None
        data = pipeline.add_features(bars_df)
        if len(data) < pipeline.SEQUENCE_LENGTH + 10:
            return None
        data = pipeline.add_sentiment(data, symbol)

        for col in pipeline.BASE_FEATURES:
            if col not in data.columns:
                data[col] = 0.0

        # ── GRU forward pass (no weight updates) ──────────────────────────
        device = _torch.device("cpu")
        with open(gru_scaler_path, 'rb') as f:
            gru_scaler = _pickle.load(f)
        with open(gru_meta_path) as f:
            gru_meta = json.load(f)

        X_all = gru_scaler.transform(data[pipeline.BASE_FEATURES].values)

        gru_model = pipeline.GRUModel(input_size=len(pipeline.BASE_FEATURES)).to(device)
        gru_model.load_state_dict(_torch.load(gru_model_path, map_location=device))
        gru_model.eval()

        seqs = _np.array([
            X_all[i - pipeline.SEQUENCE_LENGTH:i]
            for i in range(pipeline.SEQUENCE_LENGTH, len(X_all))
        ])
        with _torch.no_grad():
            raw_probs = gru_model(
                _torch.tensor(seqs, dtype=_torch.float32)
            ).squeeze().cpu().numpy()
        if raw_probs.ndim == 0:
            raw_probs = _np.array([float(raw_probs)])

        if gru_meta.get("inverted", False):
            raw_probs = 1.0 - raw_probs

        gru_prob = float(raw_probs[-1])

        # EMA-3 of the most recent gru_prob values (mirrors training behaviour)
        recent = raw_probs[-10:] if len(raw_probs) >= 10 else raw_probs
        ema_val = float(recent[0])
        alpha_ema = 2.0 / (3 + 1)
        for p in recent[1:]:
            ema_val = alpha_ema * float(p) + (1 - alpha_ema) * ema_val
        gru_prob_ema3 = ema_val

        # Attach gru columns so EXTENDED_FEATURES is fully populated
        gru_prob_col = _np.full(len(data), _np.nan)
        for i, p in enumerate(raw_probs):
            gru_prob_col[pipeline.SEQUENCE_LENGTH + i] = p
        data = data.copy()
        data['gru_prob']      = gru_prob_col
        data['gru_prob_ema3'] = data['gru_prob'].ewm(span=3, adjust=False).mean()

        for col in pipeline.EXTENDED_FEATURES:
            if col not in data.columns:
                data[col] = 0.0

        data_valid = data.dropna(subset=['gru_prob'])
        if len(data_valid) < 6:
            return None

        current_row = data_valid.iloc[-1]
        old_row     = data_valid.iloc[-6]   # ~5 business days ago

        # ── SGD: partial_fit on newly resolved 5-day label ────────────────
        with open(sgd_scaler_path, 'rb') as f:
            sgd_scaler = _pickle.load(f)
        with open(sgd_model_path, 'rb') as f:
            sgd_model = _pickle.load(f)

        true_label = int(float(current_row['Close']) > float(old_row['Close']))
        X_old = sgd_scaler.transform(
            old_row[pipeline.EXTENDED_FEATURES].values.reshape(1, -1)
        )
        try:
            sgd_model.partial_fit(X_old, [true_label], classes=[0, 1])
        except Exception:
            pass

        with open(sgd_model_path, 'wb') as f:
            _pickle.dump(sgd_model, f)

        X_cur    = sgd_scaler.transform(
            current_row[pipeline.EXTENDED_FEATURES].values.reshape(1, -1)
        )
        sgd_pred = int(sgd_model.predict(X_cur)[0])
        raw_conf = float(sgd_model.decision_function(X_cur)[0])
        sgd_conf = float(1.0 / (1.0 + _np.exp(-raw_conf)))

        # ── RL: single observation → action ───────────────────────────────
        from stable_baselines3 import PPO as _PPO
        rl_model = _PPO.load(rl_model_path)

        prev         = prev_map.get(symbol, {})
        prev_sig     = prev.get("last_signal", "FLAT")
        position     = 1 if prev_sig in ("BUY", "HOLD") else 0
        hold_streak  = 0

        rl_feats = []
        for col in pipeline.RL_FEATURES:
            if col == 'sgd_conf':
                rl_feats.append(sgd_conf)
            else:
                val = current_row.get(col, 0.0)
                rl_feats.append(float(val) if not _pd.isna(val) else 0.0)
        obs = _np.array(rl_feats + [float(position), float(hold_streak)],
                        dtype=_np.float32)
        rl_action, _ = rl_model.predict(obs, deterministic=True)
        rl_action    = int(rl_action)

        if   rl_action == 1 and position == 0: new_last_signal = "BUY"
        elif rl_action == 1 and position == 1: new_last_signal = "HOLD"
        elif rl_action == 0 and position == 1: new_last_signal = "SELL"
        else:                                  new_last_signal = "FLAT"

        current_price = float(current_row['Close'])
        row_atr = float(current_row.get('ATR_14', 0) or 0)
        row_adx = float(current_row.get('ADX_14', 25) or 25)

        row = {
            "Symbol":        symbol,
            "GRU Acc %":     prev.get("gru_accuracy", 50.0),
            "GRU Signal":    "UP  " if gru_prob >= 0.5 else "DOWN",
            "GRU Prob":      round(gru_prob, 3),
            "SGD Acc %":     prev.get("sgd_accuracy", 50.0),
            "SGD Acc-10 %":  prev.get("sgd_accuracy_10", 50.0),
            "RL Win Rate %": prev.get("rl_win_rate", 50.0),
            "Backtest $":    prev.get("backtest_profit", 0),
            "Return %":      prev.get("return_pct", 0),
            "Hold Return %": prev.get("hold_return_pct", 0),
            "Alpha %":       prev.get("alpha_pct", 0),
            "Allocation $":  prev.get("starting_allocation", 0),
            "Qty":           prev.get("quantity", 0),
            "Buys":          prev.get("n_buys", 0),
            "Holds":         prev.get("n_holds", 0),
            "Sells":         prev.get("n_sells", 0),
            "Avg Hold (wk)": prev.get("avg_hold_weeks", 0),
            "chart_data":    prev.get("chart_data"),
            "current_price": round(current_price, 2),
            "ATR_14":        row_atr,
            "ADX_14":        row_adx,
            "last_signal":   new_last_signal,
        }
        print(f"  [INFER] {symbol}: GRU={gru_prob:.3f}→{'UP' if gru_prob >= 0.5 else 'DN'}"
              f" SGD={sgd_pred} RL={'LONG' if rl_action else 'FLAT'}"
              f" sig={new_last_signal} price=${current_price:.2f}")
        return row

    except Exception as _e:
        print(f"  [INFER] {symbol} failed ({_e}), falling back to full train.")
        return None


# ── Main demo ─────────────────────────────────────────────────────────────────

banner("STOCK MARKET AI — DEMO")
print(f"  Mode          : {args.mode.upper()}")
print(f"  Profile       : {args.profile.upper()}")
print(f"  Symbols : {' '.join(args.symbols)}")
if args.mode == "train":
    print(f"  GRU epochs    : {args.epochs}  (full pipeline uses 30)")
    print(f"  RL timesteps  : {DEMO_RL_STEPS}  (full pipeline uses 8000)")

from dotenv import load_dotenv
load_dotenv()
has_finnhub = bool(os.getenv("FINNHUB_API_KEY", ""))
print(f"  Sentiment     : {'enabled (Finnhub)' if has_finnhub else 'disabled (set FINNHUB_API_KEY in .env to enable)'}")

def compute_signals(preds):
    """Convert binary predictions to BUY/HOLD/SELL/FLAT labels."""
    signals, prev = [], 0
    for p in preds:
        p = int(p)
        if   p == 1 and prev == 0: signals.append("BUY")
        elif p == 1 and prev == 1: signals.append("HOLD")
        elif p == 0 and prev == 1: signals.append("SELL")
        else:                       signals.append("FLAT")
        prev = p
    return signals

all_results = []
failed = []

for symbol in args.symbols:
    banner(f"Processing: {symbol}", char="-")
    t0 = time.time()

    # ── Fast inference path (skips retraining) ────────────────────────────
    if args.mode == "infer":
        infer_row = _try_infer(symbol, _prev_results_map)
        if infer_row is not None:
            all_results.append(infer_row)
            print(f"  Done in {time.time() - t0:.0f}s  [infer mode]")
            continue
        print(f"  Falling back to full train for {symbol}...")

    try:
        # ── Fetch & engineer features ──────────────────────────────────────
        print(f"\n  Fetching {symbol} data from Yahoo Finance...")
        bars_df = pipeline.fetch_historical_data(symbol, days=600)
        time.sleep(1)  # avoid yfinance rate limiting
        if bars_df.empty:
            print(f"  No data returned for {symbol}, skipping.")
            failed.append(symbol)
            continue

        data = pipeline.add_features(bars_df)
        min_rows = pipeline.TEST_SIZE + pipeline.SEQUENCE_LENGTH + 30
        if data.empty or len(data) < min_rows:
            print(f"  Not enough history for {symbol} (need {min_rows} rows, got {len(data)}), skipping.")
            failed.append(symbol)
            continue

        data = pipeline.add_sentiment(data, symbol)
        print(f"  Data ready: {len(data)} rows")

        # ── Stage 1: GRU ──────────────────────────────────────────────────
        section("Stage 1 / 3 — GRU (temporal pattern recognition)")
        data_with_gru, gru_acc, gru_next_prob = pipeline.run_gru_stage(data, symbol, DEMO_GRU_DIR)
        if len(data_with_gru) < pipeline.TEST_SIZE + 10:
            print(f"  Not enough post-GRU rows, skipping.")
            failed.append(symbol)
            continue

        # ── Stage 2: SGD ──────────────────────────────────────────────────
        section("Stage 2 / 3 — SGD Classifier (linear decision boundary)")
        sgd_result = pipeline.run_sgd_stage(data_with_gru, symbol, DEMO_SGD_DIR)

        # ── Stage 3: RL ───────────────────────────────────────────────────
        section("Stage 3 / 3 — PPO Reinforcement Learning (strategy)")

        # Temporarily patch RL timesteps
        import stable_baselines3
        _orig_learn = stable_baselines3.PPO.learn
        def _patched_learn(self_rl, total_timesteps, **kw):
            return _orig_learn(self_rl, DEMO_RL_STEPS, **kw)
        stable_baselines3.PPO.learn = _patched_learn

        rl_result = pipeline.run_rl_stage(data_with_gru, sgd_result, symbol, DEMO_RL_DIR)

        stable_baselines3.PPO.learn = _orig_learn  # restore

        # ── Extract last-row indicators for portfolio scoring ─────────────────
        try:
            _last = sgd_result["test_data"].iloc[-1]
            row_atr = float(_last.get('ATR_14', 0) or 0)
            row_adx = float(_last.get('ADX_14', 25) or 25)
        except Exception:
            row_atr, row_adx = 0.0, 25.0

        # ── Backtest trade stats ───────────────────────────────────────────
        current_price = sgd_result["close_prices_test"][-1]
        quantity = max(1, int((pipeline.PORTFOLIO_SIZE * pipeline.ALLOCATION_PERCENT / 100) // current_price))

        # Flip predictions when model is sub-chance (inverted signal is useful)
        predictions = sgd_result["predictions"].copy()
        sgd_acc = sgd_result["accuracy"]
        if sgd_acc <= 0.48:
            predictions = 1 - predictions
            sgd_acc = 1 - sgd_acc

        stats = pipeline.calculate_trade_stats(
            predictions,
            sgd_result["close_prices_test"],
            quantity,
        )

        # ── Buy-and-hold return over the same test window ─────────────────────
        p_test = sgd_result["close_prices_test"]
        hold_return_pct = round(
            (float(p_test[-1]) - float(p_test[0])) / float(p_test[0]) * 100, 1
        )
        alpha = round(stats["return_pct"] - hold_return_pct, 1)

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.0f}s")
        print(f"  Allocation: ${stats['starting_allocation']:,.0f} | "
              f"Profit: ${stats['profit']:+,.0f} | Return: {stats['return_pct']:+.1f}%")
        print(f"  Buy & Hold: {hold_return_pct:+.1f}%  |  Alpha: {alpha:+.1f}%")
        print(f"  Trades: {stats['n_buys']} buys, {stats['n_holds']} holds, "
              f"{stats['n_sells']} sells | Avg hold: {stats['avg_hold_weeks']}wk")

        # ── Chart data: actual prices + trade signals ──────────────────────────
        chart_data = None
        try:
            test_td = sgd_result["test_data"]
            if 'Timestamp' in test_td.columns:
                test_dates = [str(pd.Timestamp(d).date()) for d in test_td['Timestamp']]
            else:
                test_dates = list(range(len(predictions)))
            chart_data = {
                "dates":   test_dates,
                "prices":  [round(float(p), 2) for p in sgd_result["close_prices_test"]],
                "signals": compute_signals(predictions),
            }
        except Exception as ce:
            print(f"  Warning: chart data skipped — {ce}")

        row = {
            "Symbol":           symbol,
            "GRU Acc %":        round(gru_acc * 100, 1),
            "GRU Signal":       "UP  " if gru_next_prob >= 0.5 else "DOWN",
            "GRU Prob":         round(gru_next_prob, 3),
            "SGD Acc %":        round(sgd_acc * 100, 1),
            "SGD Acc-10 %":     round(sgd_result["last_10_accuracy"] * 100, 1),
            "RL Win Rate %":    round(rl_result["win_rate"] * 100, 1),
            "Backtest $":       round(stats["profit"], 0),
            "Return %":         stats["return_pct"],
            "Hold Return %":    hold_return_pct,
            "Alpha %":          alpha,
            "Allocation $":     stats["starting_allocation"],
            "Qty":              stats["quantity"],
            "Buys":             stats["n_buys"],
            "Holds":            stats["n_holds"],
            "Sells":            stats["n_sells"],
            "Avg Hold (wk)":    stats["avg_hold_weeks"],
            "chart_data":       chart_data,
            "current_price":    round(float(current_price), 2),
            "ATR_14":           row_atr,
            "ADX_14":           row_adx,
            "last_signal":      (chart_data["signals"][-1]
                                 if chart_data and chart_data.get("signals") else "FLAT"),
        }
        all_results.append(row)

    except Exception as e:
        print(f"\n  Error processing {symbol}: {e}")
        failed.append(symbol)

# ── Portfolio allocation ──────────────────────────────────────────────────────

def _compute_portfolio(results, portfolio_usd=100_000, cap_pct=25.0, max_positions=10,
                       aggressive=False):
    """
    Pick the top max_positions UP-signal stocks ranked by a combined score.

    Conservative: conf × exp_move × (1 + return_boost)
    Aggressive:   exp_move^1.5 × (1 + return_boost²) × (0.3 + 0.7×conf)
                  — heavily rewards expected weekly move (ATR-based volatility)
                    and squares the return boost so past big winners get priority
    """
    def _confidence(r):
        gru_prob = float(r["GRU Prob"])
        sgd_acc  = float(r["SGD Acc %"]) / 100.0
        adx      = float(r.get("ADX_14") or 25)
        return (0.5 * abs(gru_prob - 0.5) * 2
                + 0.3 * max(0.0, (sgd_acc - 0.5) * 2)
                + 0.2 * min(adx / 50.0, 1.0))

    # ── Score every UP-signal stock ────────────────────────────────────────
    scored = []
    for r in results:
        if r["GRU Signal"].strip() != "UP":
            continue
        gru_prob     = float(r["GRU Prob"])
        atr          = float(r.get("ATR_14") or 0)
        price        = float(r["current_price"])
        conf         = _confidence(r)
        exp_move     = (atr / price * 100) if price > 0 else 0
        return_pct   = float(r.get("Return %") or 0)
        hold_ret_pct = float(r.get("Hold Return %") or 0)
        return_boost = max(return_pct, 0) / 100.0
        if aggressive:
            # Confidence stays a hard multiplier; volatility and return both
            # amplified vs conservative but can't override a weak signal
            raw_score = conf * (max(exp_move, 0.5) ** 1.3) * (1 + return_boost ** 1.5)
        else:
            raw_score = conf * max(exp_move, 0.5) * (1 + return_boost)
        scored.append({
            "symbol":            r["Symbol"],
            "signal":            "UP",
            "gru_prob":          round(gru_prob, 3),
            "confidence":        round(conf, 3),
            "expected_move_pct": round(exp_move, 2),
            "return_pct":        round(return_pct, 1),
            "hold_return_pct":   round(hold_ret_pct, 1),
            "raw_score":         raw_score,
        })

    # ── Keep only top max_positions by combined score ──────────────────────
    scored.sort(key=lambda x: x["raw_score"], reverse=True)
    candidates = scored[:max_positions]

    # ── Normalize to 100%, iterative 25%-cap redistribution ───────────────
    allocs = {}
    if candidates:
        total_score = sum(c["raw_score"] for c in candidates)
        allocs = {c["symbol"]: (c["raw_score"] / total_score * 100) if total_score > 0
                  else (100.0 / len(candidates)) for c in candidates}
        for _ in range(20):
            capped = {k: v for k, v in allocs.items() if v > cap_pct}
            if not capped:
                break
            excess = sum(v - cap_pct for v in capped.values())
            for k in capped:
                allocs[k] = cap_pct
            uncapped = {k: v for k, v in allocs.items() if v < cap_pct}
            if not uncapped:
                break
            unc_total = sum(uncapped.values())
            if unc_total == 0:
                break
            for k in uncapped:
                allocs[k] += (allocs[k] / unc_total) * excess

    invested_pct = sum(allocs.values())
    positions = sorted(
        [{
            **{k: v for k, v in c.items() if k != "raw_score"},
            "allocation_pct": round(allocs.get(c["symbol"], 0), 1),
            "allocation_usd": int(round(portfolio_usd * allocs.get(c["symbol"], 0) / 100)),
        } for c in candidates],
        key=lambda x: x["allocation_pct"], reverse=True,
    )

    # ── Per-stock action labels (all stocks, not just UP) ─────────────────
    # last_signal tells us what the model was doing at end of the backtest
    all_actions = []
    for r in results:
        last_sig   = r.get("last_signal", "FLAT")
        was_holding = last_sig in ("BUY", "HOLD")
        is_bullish  = r["GRU Signal"].strip() == "UP"

        if   was_holding and is_bullish:     action = "HOLD"
        elif was_holding and not is_bullish: action = "EXIT"
        elif not was_holding and is_bullish: action = "ENTER"
        else:                                action = "WAIT"

        all_actions.append({
            "symbol":      r["Symbol"],
            "action":      action,
            "was_holding": was_holding,
            "gru_signal":  r["GRU Signal"].strip(),
            "confidence":  round(_confidence(r), 3),
            "return_pct":  r.get("Return %"),
            "hold_return_pct": r.get("Hold Return %"),
            "alpha_pct":   r.get("Alpha %"),
        })

    # Sort: HOLD first, ENTER, EXIT, WAIT
    order = {"HOLD": 0, "ENTER": 1, "EXIT": 2, "WAIT": 3}
    all_actions.sort(key=lambda x: order.get(x["action"], 4))

    # ── Weighted portfolio backtest return ────────────────────────────────
    portfolio_return_pct = sum(
        (allocs.get(r["Symbol"], 0) / 100) * (r.get("Return %") or 0)
        for r in results if r["Symbol"] in allocs
    )

    # Determine backtest date range from first available chart_data
    test_start, test_end, test_days = None, None, 0
    for r in results:
        cd = r.get("chart_data")
        if cd and cd.get("dates") and len(cd["dates"]) > 1:
            test_start = cd["dates"][0]
            test_end   = cd["dates"][-1]
            test_days  = len(cd["dates"])
            break

    return {
        "positions":              positions,
        "all_actions":            all_actions,
        "cash_pct":               round(max(0.0, 100.0 - invested_pct), 1),
        "n_positions":            len(positions),
        "portfolio_return_pct":   round(portfolio_return_pct, 1),
        "portfolio_value_usd":    int(round(portfolio_usd * (1 + portfolio_return_pct / 100))),
        "test_start":             test_start,
        "test_end":               test_end,
        "test_days":              test_days,
    }


portfolio_data = _compute_portfolio(
    all_results,
    max_positions=PROFILE_MAX_POS,
    cap_pct=PROFILE_CAP_PCT,
    aggressive=(args.profile == "aggressive"),
)

# ── Summary ───────────────────────────────────────────────────────────────────

banner("DEMO RESULTS SUMMARY")

if all_results:
    df = pd.DataFrame(all_results)

    col_widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}
    header = "  " + "  ".join(col.ljust(col_widths[col]) for col in df.columns)
    divider = "  " + "  ".join("-" * col_widths[col] for col in df.columns)
    print(header)
    print(divider)
    for _, row in df.iterrows():
        print("  " + "  ".join(str(row[col]).ljust(col_widths[col]) for col in df.columns))

    output = "demo_results.xlsx"
    try:
        df.to_excel(output, index=False)
        print(f"\n  Results saved to {output}")
    except Exception as e:
        print(f"\n  Could not save Excel file: {e}")

    # ── Benchmark returns over the same test window ───────────────────────────
    def _fetch_benchmark_return(ticker):
        try:
            df = pipeline.fetch_historical_data(ticker, days=400)
            if df.empty or len(df) < pipeline.TEST_SIZE:
                return None
            test  = df.iloc[-pipeline.TEST_SIZE:]
            start = float(test["Close"].iloc[0])
            end   = float(test["Close"].iloc[-1])
            return round((end - start) / start * 100, 1)
        except Exception:
            return None

    print("\n  Fetching benchmarks (SPY, VOO, SPMO)...")
    spy_return_pct  = _fetch_benchmark_return("SPY")
    voo_return_pct  = _fetch_benchmark_return("VOO")
    spmo_return_pct = _fetch_benchmark_return("SPMO")
    for name, ret in [("SPY", spy_return_pct), ("VOO", voo_return_pct), ("SPMO", spmo_return_pct)]:
        print(f"  {name}: {f'{ret:+.1f}%' if ret is not None else 'unavailable'}")
    portfolio_data["benchmarks"] = {"VOO": voo_return_pct, "SPMO": spmo_return_pct}

    # ── Equity curve (model vs SPMO vs VOO over test window) ─────────────────
    def _compute_equity_curve(positions, test_size=pipeline.TEST_SIZE):
        try:
            weights = {p["symbol"]: p["allocation_pct"] / 100.0 for p in positions}
            bench_syms = ["SPMO", "VOO"]
            all_syms = list(weights.keys()) + bench_syms
            prices = {}
            for sym in all_syms:
                try:
                    df_p = pipeline.fetch_historical_data(sym, days=test_size + 30)
                    if not df_p.empty and len(df_p) >= test_size:
                        prices[sym] = df_p["Close"].iloc[-test_size:].reset_index(drop=True)
                except Exception:
                    pass
            if not prices:
                return None
            ref = prices.get("SPMO") or prices.get("VOO") or next(iter(prices.values()))
            n = len(ref)
            if n < 5:
                return None
            # Use SPY dates as labels if available, else sequential
            try:
                df_dates = pipeline.fetch_historical_data("SPY", days=test_size + 30)
                date_labels = [str(d.date()) for d in df_dates.index[-n:]]
            except Exception:
                date_labels = list(range(n))

            def norm(series):
                base = float(series.iloc[0])
                return [round(float(v) / base * 100, 2) for v in series] if base else [100.0] * n

            total_w = sum(weights.values())
            model_curve = [100.0] * n
            if total_w > 0:
                for sym, w in weights.items():
                    if sym in prices and len(prices[sym]) == n:
                        s = norm(prices[sym])
                        model_curve = [model_curve[i] + (s[i] - 100) * (w / total_w) for i in range(n)]

            result = {"dates": date_labels, "model": [round(v, 2) for v in model_curve]}
            for b in bench_syms:
                if b in prices and len(prices[b]) == n:
                    result[b] = norm(prices[b])
            return result
        except Exception as e:
            print(f"  Could not compute equity curve: {e}")
            return None

    print("\n  Computing equity curve...")
    equity_curve = _compute_equity_curve(portfolio_data.get("positions", []))
    if equity_curve:
        portfolio_data["equity_curve"] = equity_curve
        print(f"  Equity curve: {len(equity_curve['dates'])} data points")

    # ── Forward prediction tracking ───────────────────────────────────────────
    import math
    from collections import defaultdict
    from pandas.tseries.offsets import BDay

    pred_log_path = os.path.join("web", "data", "predictions_log.json")
    today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    today_ts  = pd.Timestamp(datetime.datetime.utcnow()).normalize()

    pred_log = {"predictions": []}
    if os.path.exists(pred_log_path):
        try:
            with open(pred_log_path) as pf:
                pred_log = json.load(pf)
        except Exception:
            pred_log = {"predictions": []}

    # Resolve pending predictions that are 5+ business days old
    for rec in pred_log.get("predictions", []):
        if rec.get("correct") is not None or rec.get("price_at_prediction") is None:
            continue
        try:
            outcome_ts = pd.Timestamp(rec["run_date"]) + 5 * BDay()
            if today_ts < outcome_ts:
                continue
            hist = yf.Ticker(rec["symbol"]).history(
                start=outcome_ts.strftime("%Y-%m-%d"),
                end=(outcome_ts + 3 * BDay()).strftime("%Y-%m-%d"),
            )
            if hist.empty:
                continue
            outcome_price = float(hist["Close"].iloc[0])
            actual_up     = outcome_price > rec["price_at_prediction"]
            predicted_up  = rec["gru_signal"] == "UP"
            rec["outcome_date"]  = outcome_ts.strftime("%Y-%m-%d")
            rec["outcome_price"] = round(outcome_price, 2)
            rec["correct"]       = (actual_up == predicted_up)
        except Exception as ex:
            print(f"  Warning: outcome lookup failed for {rec.get('symbol','?')}: {ex}")

    # Drop records older than 180 days
    cutoff = (today_ts - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    pred_log["predictions"] = [
        r for r in pred_log.get("predictions", [])
        if r.get("run_date", "") >= cutoff
    ]

    # Append today's predictions (skip symbols already logged today)
    logged_today = {r["symbol"] for r in pred_log["predictions"] if r.get("run_date") == today_str}
    for r in all_results:
        if r["Symbol"] in logged_today:
            continue
        pred_log["predictions"].append({
            "run_date":            today_str,
            "symbol":              r["Symbol"],
            "gru_signal":          r["GRU Signal"].strip(),
            "gru_prob":            r["GRU Prob"],
            "price_at_prediction": r["current_price"],
            "outcome_date":        None,
            "outcome_price":       None,
            "correct":             None,
        })

    pred_log["last_updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        os.makedirs(os.path.dirname(pred_log_path), exist_ok=True)
        with open(pred_log_path, "w") as pf:
            json.dump(pred_log, pf, indent=2)
        print(f"  Prediction log updated ({len(pred_log['predictions'])} entries) → {pred_log_path}")
    except Exception as ex:
        print(f"  Could not save prediction log: {ex}")

    # Per-symbol live accuracy summary (resolved predictions only)
    sym_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for rec in pred_log.get("predictions", []):
        if rec.get("correct") is None:
            continue
        sym_stats[rec["symbol"]]["total"] += 1
        if rec["correct"]:
            sym_stats[rec["symbol"]]["correct"] += 1

    prediction_summary = {
        sym: {
            "live_accuracy":  round(v["correct"] / v["total"] * 100, 1),
            "n_correct":      v["correct"],
            "n_resolved":     v["total"],
        }
        for sym, v in sym_stats.items()
    }

    # ── Live portfolio P&L (reads trades_log.json, fetches current prices) ───────
    trades_log_path = os.path.join("web", "data", "trades_log.json")
    live_portfolio = {
        "positions": [], "total_cost": 0, "total_value": 0,
        "total_pnl_usd": 0, "total_pnl_pct": 0, "n_positions": 0, "as_of": None,
    }
    try:
        if os.path.exists(trades_log_path):
            with open(trades_log_path) as tlf:
                trades_data = json.load(tlf)
            open_trades = [t for t in trades_data.get("trades", []) if t.get("status") == "open"]
            if open_trades:
                syms = list({t["symbol"] for t in open_trades})
                print(f"\n  Fetching live prices for {len(syms)} held positions...")
                prices = {}
                for sym in syms:
                    try:
                        df = yf.Ticker(sym).history(period="2d")
                        if not df.empty:
                            prices[sym] = round(float(df["Close"].iloc[-1]), 2)
                    except Exception:
                        pass
                positions_pnl = []
                total_cost = total_value = 0.0
                for t in open_trades:
                    sym          = t["symbol"]
                    shares       = float(t.get("shares", 0))
                    entry_price  = float(t.get("entry_price", 0))
                    current_price = prices.get(sym, entry_price)
                    cost  = shares * entry_price
                    value = shares * current_price
                    pnl   = value - cost
                    pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
                    total_cost  += cost
                    total_value += value
                    try:
                        days_held = (datetime.date.today() - datetime.date.fromisoformat(t.get("entry_date", str(datetime.date.today())))).days
                    except Exception:
                        days_held = 0
                    positions_pnl.append({
                        "symbol":        sym,
                        "shares":        round(shares, 6),
                        "entry_price":   round(entry_price, 2),
                        "current_price": round(current_price, 2),
                        "entry_date":    t.get("entry_date"),
                        "days_held":     days_held,
                        "cost_basis":    round(cost, 2),
                        "current_value": round(value, 2),
                        "pnl_usd":       round(pnl, 2),
                        "pnl_pct":       round(pnl_pct, 2),
                    })
                positions_pnl.sort(key=lambda x: x["pnl_pct"], reverse=True)
                total_pnl = total_value - total_cost
                live_portfolio = {
                    "positions":     positions_pnl,
                    "total_cost":    round(total_cost, 2),
                    "total_value":   round(total_value, 2),
                    "total_pnl_usd": round(total_pnl, 2),
                    "total_pnl_pct": round(total_pnl / total_cost * 100 if total_cost > 0 else 0, 2),
                    "n_positions":   len(positions_pnl),
                    "as_of":         datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                print(f"  Live portfolio: {len(positions_pnl)} positions · "
                      f"value=${total_value:,.0f} · P&L=${total_pnl:+,.0f} "
                      f"({live_portfolio['total_pnl_pct']:+.1f}%)")
    except Exception as e:
        print(f"  Live portfolio P&L skipped: {e}")

    # ── Write JSON for Vercel dashboard ───────────────────────────────────────

    def _safe(v):
        """Recursively convert numpy types and replace NaN/Inf with None."""
        if v is None:
            return None
        if isinstance(v, dict):
            return {k: _safe(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        try:
            import numpy as np
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                v = float(v)
        except ImportError:
            pass
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    json_path = RESULTS_PATH
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        json_records = []
        for r in all_results:
            json_records.append(_safe({
                "symbol":             r["Symbol"],
                "gru_accuracy":       r["GRU Acc %"],
                "gru_signal":         r["GRU Signal"].strip(),
                "gru_prob":           r["GRU Prob"],
                "sgd_accuracy":       r["SGD Acc %"],
                "sgd_accuracy_10":    r["SGD Acc-10 %"],
                "rl_win_rate":        r["RL Win Rate %"],
                "backtest_profit":    r["Backtest $"],
                "return_pct":         r["Return %"],
                "starting_allocation": r["Allocation $"],
                "quantity":           r["Qty"],
                "n_buys":             r["Buys"],
                "n_holds":            r["Holds"],
                "n_sells":            r["Sells"],
                "avg_hold_weeks":     r["Avg Hold (wk)"],
                "hold_return_pct":    r["Hold Return %"],
                "alpha_pct":          r["Alpha %"],
                "chart_data":         r.get("chart_data"),
            }))
        payload = _safe({
            "last_updated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_config": {
                "symbols": args.symbols,
                "epochs":   args.epochs,
                "rl_steps": DEMO_RL_STEPS,
                "profile":  args.profile,
            },
            "spy_return_pct":      spy_return_pct,
            "prediction_summary":  prediction_summary,
            "portfolio":           portfolio_data,
            "live_portfolio":      live_portfolio,
            "recent_predictions":  sorted(
                pred_log.get("predictions", []),
                key=lambda x: x.get("run_date", ""),
                reverse=True,
            )[:40],
            "results": json_records,
        })
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Dashboard data saved to {json_path}")
    except Exception as e:
        print(f"  Could not save dashboard JSON: {e}")
else:
    print("  No results to display.")

if failed:
    print(f"\n  Skipped: {', '.join(failed)}")

print("""
  Notes:
  - Demo models are saved in Demo_Models/ and won't overwrite your trained models.
  - Accuracy reflects the last {test_size} days of data, not live performance.
  - GRU Signal is the model's prediction for end-of-next-week direction.
  - To run the full pipeline on all S&P 500 stocks: python Combined_Model.py
""".format(test_size=pipeline.TEST_SIZE))
