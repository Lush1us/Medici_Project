"""
Medici — Market Data Fetcher

Pulls EOD OHLCV data via yfinance and stores as parquet files.
One file per ticker in data/eod/
"""

import yfinance as yf
import os
import sys
import json
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")

# Core universe — SPY + major sectors + a few high-volume singles
TICKERS = [
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000
    "DIA",    # Dow 30
    "TLT",    # 20+ Year Treasury (rates context)
    "GLD",    # Gold
    "USO",    # Oil
    "VIX",    # Volatility index (via ^VIX)
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
]

# VIX needs the ^ prefix for yfinance
TICKER_MAP = {"VIX": "^VIX"}

PERIOD = "2y"  # 2 years of daily data


def fetch_ticker(ticker):
    yf_ticker = TICKER_MAP.get(ticker, ticker)
    try:
        df = yf.download(yf_ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            print(f"  {ticker}: NO DATA")
            return None

        # Flatten multi-level columns if present
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)

        out_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
        df.to_parquet(out_path)
        print(f"  {ticker}: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")
        return {"ticker": ticker, "rows": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date())}
    except Exception as e:
        print(f"  {ticker}: ERROR — {e}")
        return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching {len(TICKERS)} tickers ({PERIOD} daily data)")
    print(f"Output: {DATA_DIR}/\n")

    results = []
    for t in TICKERS:
        r = fetch_ticker(t)
        if r:
            results.append(r)

    # Write manifest
    manifest = {
        "fetched_at": datetime.now().isoformat(),
        "period": PERIOD,
        "tickers": results,
    }
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(results)}/{len(TICKERS)} tickers saved.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
