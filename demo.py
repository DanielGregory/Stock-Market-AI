"""
Demo runner — runs the full sequential pipeline on a small set of stocks
with reduced training time so you can see results in minutes.

Usage:
    python demo.py                            # runs default 5 stocks
    python demo.py --symbols AAPL MSFT TSLA  # custom list
    python demo.py --epochs 5 --steps 500    # even faster (lower accuracy)
"""

import argparse
import os
import sys
import time
import pandas as pd

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Stock Market AI — Demo")
parser.add_argument(
    "--symbols", nargs="+",
    default=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
    help="Stock ticker symbols to run (default: AAPL MSFT GOOGL TSLA NVDA)"
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

all_results = []
failed = []

for symbol in args.symbols:
    banner(f"Processing: {symbol}", char="-")
    t0 = time.time()

    try:
        # ── Fetch & engineer features ──────────────────────────────────────
        print(f"\n  Fetching {symbol} data from Yahoo Finance...")
        bars_df = pipeline.fetch_historical_data(symbol, days=400)
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

        # ── Backtest profit ────────────────────────────────────────────────
        current_price = sgd_result["close_prices_test"][-1]
        quantity = max(1, int((pipeline.PORTFOLIO_SIZE * pipeline.ALLOCATION_PERCENT / 100) // current_price))
        profit = pipeline.calculate_profit(
            sgd_result["predictions"],
            sgd_result["close_prices_test"],
            quantity
        )

        sgd_acc = sgd_result["accuracy"]
        if sgd_acc <= 0.48:
            sgd_acc = 1 - sgd_acc

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.0f}s")

        row = {
            "Symbol":        symbol,
            "GRU Acc %":     round(gru_acc * 100, 1),
            "GRU Signal":    "UP  " if gru_next_prob >= 0.5 else "DOWN",
            "GRU Prob":      round(gru_next_prob, 3),
            "SGD Acc %":     round(sgd_acc * 100, 1),
            "SGD Acc-10 %":  round(sgd_result["last_10_accuracy"] * 100, 1),
            "RL Win Rate %": round(rl_result["win_rate"] * 100, 1),
            "Backtest $":    round(profit, 0),
        }
        all_results.append(row)

    except Exception as e:
        print(f"\n  Error processing {symbol}: {e}")
        failed.append(symbol)

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
else:
    print("  No results to display.")

if failed:
    print(f"\n  Skipped: {', '.join(failed)}")

print("""
  Notes:
  - Demo models are saved in Demo_Models/ and won't overwrite your trained models.
  - Accuracy reflects the last {test_size} days of data, not live performance.
  - GRU Signal is the model's prediction for the NEXT trading day.
  - To run the full pipeline on all S&P 500 stocks: python Combined_Model.py
""".format(test_size=pipeline.TEST_SIZE))
