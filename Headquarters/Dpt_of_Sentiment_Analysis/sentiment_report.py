"""
Department of Sentiment Analysis — Daily Report

Placeholder. Needs external data sources:
- Options flow (unusual activity, put/call ratios)
- Insider transactions (SEC Form 4)
- News sentiment scoring
- Social media signals

For now, returns a basic volume-derived sentiment proxy.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "eod")


def _load(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def compute_sentiment(ticker: str, as_of_date: str) -> dict | None:
    """Basic sentiment proxy from price/volume action. Not real sentiment."""
    df = _load(ticker)
    if df is None:
        return None
    df = df.loc[:as_of_date]
    if len(df) < 20:
        return None

    close = df["Close"]
    volume = df["Volume"]
    c = float(close.iloc[-1])

    # Up/down volume ratio as sentiment proxy
    changes = close.diff()
    up_vol = float(volume.where(changes > 0, 0).tail(10).sum())
    down_vol = float(volume.where(changes < 0, 0).tail(10).sum())
    vol_ratio = up_vol / down_vol if down_vol > 0 else 2.0

    # Simple sentiment score: -10 to 10
    # Based on: price momentum, volume direction, recent range
    ret_5d = (c / float(close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
    ret_10d = (c / float(close.iloc[-10]) - 1) * 100 if len(close) >= 10 else 0

    sentiment = 0.0
    sentiment += min(max(ret_5d, -5), 5)                    # Short-term momentum
    sentiment += min(max((vol_ratio - 1) * 3, -3), 3)       # Volume direction
    sentiment += min(max(ret_10d / 2, -2), 2)               # Medium-term momentum
    sentiment = round(max(min(sentiment, 10), -10), 1)

    return {
        "ticker": ticker,
        "date": as_of_date,
        "sentiment_score": sentiment,
        "catalysts": [],  # TODO: external data
        "flow_signals": [],  # TODO: options flow, insider buys
        "summary": f"{ticker}: Sentiment {sentiment:+.1f} (volume-proxy only). Up/Down vol ratio {vol_ratio:.1f}. No external data sources connected.",
    }


def run_daily(ticker: str, as_of_date: str) -> dict | None:
    return compute_sentiment(ticker, as_of_date)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    date = sys.argv[2] if len(sys.argv) > 2 else "2025-03-10"
    result = run_daily(ticker, date)
    if result:
        print(json.dumps(result, indent=2))
