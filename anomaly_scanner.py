"""
Medici — Anomaly Scanner

Runs at the start of the daily pipeline. Catches stocks outside the watchlist
that had abnormal moves (gaps, volume explosions) and injects them for a quick vote.

Pure Python, no LLM.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")


def _load_ticker(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_sp500_tickers() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "data", "sp500_tickers.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [t.strip() for t in f if t.strip()]


def scan_anomalies(as_of_date: str, watchlist_tickers: set[str]) -> list[dict]:
    """Scan S&P 500 for outsiders with abnormal activity on as_of_date.

    Triggers (any one):
      - Gap > 3% (open vs prev close)
      - RVOL > 3.0 (volume vs 20-day avg)
      - Daily range > 2.5x the 20-day avg range

    Returns list of anomaly dicts, sorted by severity.
    """
    tickers = _load_sp500_tickers()
    anomalies = []

    for ticker in tickers:
        if ticker in watchlist_tickers:
            continue  # already on watchlist

        df = _load_ticker(ticker)
        if df is None:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 25:
            continue

        today = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(today["Close"])
        prev_close = float(prev["Close"])
        open_ = float(today["Open"])
        volume = float(today["Volume"])
        high = float(today["High"])
        low = float(today["Low"])

        # Liquidity check
        avg_vol = float(df["Volume"].tail(20).mean())
        if avg_vol < 500_000 or close < 5.0:
            continue

        triggers = []
        severity = 0.0

        # Gap check
        gap_pct = (open_ / prev_close - 1) * 100
        if abs(gap_pct) > 3.0:
            triggers.append(f"gap {gap_pct:+.1f}%")
            severity += abs(gap_pct) / 3.0

        # RVOL check
        rvol = volume / avg_vol if avg_vol > 0 else 1.0
        if rvol > 3.0:
            triggers.append(f"RVOL {rvol:.1f}x")
            severity += rvol / 3.0

        # Range expansion check
        day_range = high - low
        avg_range = float((df["High"] - df["Low"]).tail(20).mean())
        range_ratio = day_range / avg_range if avg_range > 0 else 1.0
        if range_ratio > 2.5:
            triggers.append(f"range {range_ratio:.1f}x")
            severity += range_ratio / 2.5

        if triggers:
            daily_change = round((close / prev_close - 1) * 100, 2)
            anomalies.append({
                "ticker": ticker,
                "close": round(close, 2),
                "daily_change_pct": daily_change,
                "gap_pct": round(gap_pct, 2),
                "rvol": round(rvol, 1),
                "range_ratio": round(range_ratio, 1),
                "triggers": triggers,
                "severity": round(severity, 2),
            })

    anomalies.sort(key=lambda x: x["severity"], reverse=True)
    return anomalies


if __name__ == "__main__":
    import sys
    import json
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-03-14"
    watchlist = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else set()
    results = scan_anomalies(date, watchlist)
    print(f"Found {len(results)} anomalies for {date}")
    for r in results[:15]:
        print(f"  {r['ticker']:6s} {r['daily_change_pct']:+6.1f}% | {', '.join(r['triggers'])} | severity={r['severity']}")
