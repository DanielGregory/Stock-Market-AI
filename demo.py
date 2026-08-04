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

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Stock Market AI — Demo")
parser.add_argument(
    "--symbols", nargs="+",
    default=[
        # Core focused list — bull-run capture stocks
        "AMD", "MU", "CRWD", "NVDA", "PLTR", "COIN", "NFLX", "AAPL",
        "GLW", "NBIS", "SPMO",
        # Commented out for faster runs — re-enable to expand coverage
        # "MSFT", "GOOGL", "AMZN", "META",
        # "AVGO", "TSLA", "CRM",
        # "JPM", "GS", "LLY", "BA", "UBER",
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
args = parser.parse_args()

# ── Patch hyperparameters before importing the pipeline ───────────────────────
# This overrides the module-level constants in Combined_Model so the demo
# runs faster without changing the actual training files.

import Combined_Model as pipeline

pipeline.GRU_EPOCHS = args.epochs
pipeline.BATCH_SIZE = 16  # smaller batches for small datasets

DEMO_RL_STEPS = args.steps
DEMO_GRU_DIR = "Demo_Models/GRU"
DEMO_SGD_DIR = "Demo_Models/SGD"
DEMO_RL_DIR  = "Demo_Models/RL"

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(text, width=60, char="="):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

def section(text):
    print(f"\n  {'─' * 50}")
    print(f"  {text}")
    print(f"  {'─' * 50}")

# ── Main demo ─────────────────────────────────────────────────────────────────

banner("STOCK MARKET AI — DEMO")
print(f"  Symbols : {' '.join(args.symbols)}")
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

def _compute_portfolio(results, portfolio_usd=100_000, cap_pct=25.0):
    """
    Score UP-signal stocks by confidence × expected_move, cap at cap_pct%.
    Also produces per-stock action labels (HOLD/ENTER/EXIT/WAIT) and a
    weighted historical portfolio return from the backtest period.
    """
    def _confidence(r):
        gru_prob = float(r["GRU Prob"])
        sgd_acc  = float(r["SGD Acc %"]) / 100.0
        adx      = float(r.get("ADX_14") or 25)
        return (0.5 * abs(gru_prob - 0.5) * 2
                + 0.3 * max(0.0, (sgd_acc - 0.5) * 2)
                + 0.2 * min(adx / 50.0, 1.0))

    # ── Build candidates for allocation (UP signals only) ──────────────────
    candidates = []
    for r in results:
        if r["GRU Signal"].strip() != "UP":
            continue
        gru_prob  = float(r["GRU Prob"])
        atr       = float(r.get("ATR_14") or 0)
        price     = float(r["current_price"])
        conf      = _confidence(r)
        exp_move  = (atr / price * 100) if price > 0 else 0
        raw_score = conf * max(exp_move, 0.5)
        candidates.append({
            "symbol":            r["Symbol"],
            "signal":            "UP",
            "gru_prob":          round(gru_prob, 3),
            "confidence":        round(conf, 3),
            "expected_move_pct": round(exp_move, 2),
            "raw_score":         raw_score,
        })

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


portfolio_data = _compute_portfolio(all_results)

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

    # ── SPY baseline (buy-and-hold over the same test period) ─────────────────
    spy_return_pct = None
    try:
        print("\n  Fetching SPY baseline...")
        spy_df = pipeline.fetch_historical_data("SPY", days=400)
        if not spy_df.empty and len(spy_df) >= pipeline.TEST_SIZE:
            spy_test = spy_df.iloc[-pipeline.TEST_SIZE:]
            spy_start = float(spy_test["Close"].iloc[0])
            spy_end   = float(spy_test["Close"].iloc[-1])
            spy_return_pct = round((spy_end - spy_start) / spy_start * 100, 1)
            print(f"  SPY return over test period: {spy_return_pct:+.1f}%")
        else:
            print("  SPY data unavailable, skipping baseline.")
    except Exception as e:
        print(f"  SPY fetch failed: {e}")

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

    json_path = os.path.join("web", "data", "results.json")
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
            },
            "spy_return_pct":      spy_return_pct,
            "prediction_summary":  prediction_summary,
            "portfolio":           portfolio_data,
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
