"""
Fetch real market data and export it for the static portfolio optimizer app.

Run this locally (where you have normal internet access — yfinance cannot
be reached from within some sandboxed environments, but works fine on a
regular laptop). It writes portfolio_data.json next to index.html; the
web app automatically detects and uses it instead of its synthetic
fallback data.

Usage:
    python fetch_data_for_app.py
    python fetch_data_for_app.py --tickers AAPL MSFT GOOGL --start 2022-01-01
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from portfolio_optimizer.markowitz import annualized_stats  # noqa: E402
from shared.data_utils import fetch_prices  # noqa: E402

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "XOM", "JNJ", "PG", "GLD", "TLT", "SPY",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--out", default=str(Path(__file__).parent / "web_app" / "portfolio_data.json"))
    args = parser.parse_args()

    print(f"Fetching {len(args.tickers)} tickers from {args.start}...")
    prices = fetch_prices(args.tickers, start=args.start)
    mu, cov = annualized_stats(prices)

    data = {
        "asOf": date.today().isoformat(),
        "tickers": list(prices.columns),
        "mu": mu.values.tolist(),
        "cov": cov.values.tolist(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {out_path}")
    print("Commit this file alongside index.html and redeploy — "
          "the web app will pick it up automatically and show a "
          "'Live data' badge instead of 'Sample data'.")


if __name__ == "__main__":
    main()
