"""
Export predictions to Portfolio Analyzer — reads the latest pipeline output
(Combined_results.xlsx, falling back to demo_results.xlsx) and pushes each
stock's GRU direction/confidence to the portfolio_analyzer app's ML
prediction cache, so /api/opportunities can blend it into its score.

This is meant to run on its own schedule (cron, a local machine, or any
process outside Vercel) *after* a pipeline run finishes — never inside the
Vercel function itself. See "ML Signal Integration, Option A" in the
portfolio_analyzer expansion spec.

Usage:
    python export_predictions.py
    python export_predictions.py --url https://your-deployment.vercel.app
    python export_predictions.py --file demo_results.xlsx
"""

import argparse
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Export stock-market-ai predictions to Portfolio Analyzer")
parser.add_argument(
    "--url", default=os.environ.get("PORTFOLIO_ANALYZER_URL", "http://localhost:8000"),
    help="Base URL of the portfolio_analyzer deployment (default: $PORTFOLIO_ANALYZER_URL or localhost:8000)"
)
parser.add_argument(
    "--file", default=None,
    help="Results file to read (default: Combined_results.xlsx, falling back to demo_results.xlsx)"
)
args = parser.parse_args()

results_file = args.file
if not results_file:
    results_file = "Combined_results.xlsx" if os.path.exists("Combined_results.xlsx") else "demo_results.xlsx"

if not os.path.exists(results_file):
    print(f"No results file found ({results_file}). Run Combined_Model.py or demo.py first.")
    sys.exit(1)

df = pd.read_excel(results_file)
if "Symbol" not in df.columns or "GRU_Next_Prob" not in df.columns:
    print(f"{results_file} doesn't look like pipeline output (missing Symbol/GRU_Next_Prob columns).")
    sys.exit(1)

predictions = []
for _, row in df.iterrows():
    prob = float(row["GRU_Next_Prob"])
    direction = "up" if prob >= 0.5 else "down"
    confidence = prob if direction == "up" else (1 - prob)
    predictions.append({
        "ticker": str(row["Symbol"]).upper(),
        "direction": direction,
        "confidence": round(confidence, 4),
        "horizon": "1d",
        "model": "stock-market-ai",
    })

resp = requests.post(f"{args.url}/api/ml-predictions/ingest", json={"predictions": predictions}, timeout=30)
resp.raise_for_status()
print(f"Pushed {len(predictions)} predictions from {results_file} to {args.url} — {resp.json()}")
