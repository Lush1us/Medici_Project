"""
Medici — Phase 1: Python Screener

Pure math, no LLM. Scans S&P 500 for tickers passing regime filters.
Outputs a compact JSON array of candidates with 5 core metrics each.
"""

import os
import numpy as np
import pandas as pd
import ta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")


def load_ticker(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def load_sp500_tickers() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "data", "sp500_tickers.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [t.strip() for t in f if t.strip()]


def screen_ticker(df: pd.DataFrame, as_of_date: str) -> dict | None:
    """Compute 5 core metrics for a single ticker. Returns None if filtered out."""
    df = df.loc[:as_of_date]
    if len(df) < 200:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_close = float(close.iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1])
    sma_50 = float(close.rolling(50).mean().iloc[-1])

    # 1. ADX — trend strength
    adx_obj = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = adx_obj.adx().iloc[-1]
    if pd.isna(adx):
        return None
    adx = float(adx)

    # 2. RSI
    rsi = ta.momentum.rsi(close, window=14).iloc[-1]
    if pd.isna(rsi):
        return None
    rsi = float(rsi)

    # 3. Price vs 200-SMA (%)
    price_vs_200 = round((last_close / sma_200 - 1) * 100, 2)

    # 4. RVOL (vs 20-day avg)
    avg_vol = float(volume.rolling(20).mean().iloc[-1])
    rvol = round(float(volume.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0

    # 5. Z-Score vs 50-SMA
    std_50 = float(close.rolling(50).std().iloc[-1])
    zscore_50 = round((last_close - sma_50) / std_50, 2) if std_50 > 0 else 0.0

    # --- Regime filters: pass if ANY of these are true ---
    # Strong trend (momentum players: Giuliano, Cosimo)
    strong_trend = adx > 25
    # Overextended (mean-reversion: Piero, Lorenzo)
    overextended = abs(zscore_50) > 1.5
    # Volume spike (breakout candidates: Giuliano, Caterina)
    vol_spike = rvol > 1.5
    # RSI extreme (Piero fades, Cosimo/Lorenzo entries)
    rsi_extreme = rsi < 30 or rsi > 70
    # Big deviation from 200-SMA (Lorenzo sniper setups)
    far_from_mean = abs(price_vs_200) > 10

    if not (strong_trend or overextended or vol_spike or rsi_extreme or far_from_mean):
        return None

    return {
        "close": round(last_close, 2),
        "adx": round(adx, 1),
        "rsi": round(rsi, 1),
        "price_vs_200sma_pct": price_vs_200,
        "rvol": rvol,
        "zscore_50": zscore_50,
    }


def run_screener(as_of_date: str, spy_df: pd.DataFrame = None) -> list[dict]:
    """Screen all S&P 500 tickers. Returns list of {ticker, close, adx, rsi, ...}."""
    tickers = load_sp500_tickers()
    if not tickers:
        return []

    candidates = []
    for ticker in tickers:
        df = load_ticker(ticker)
        if df is None:
            continue
        result = screen_ticker(df, as_of_date)
        if result is not None:
            result["ticker"] = ticker
            candidates.append(result)

    # Sort by a composite "interestingness" score: higher ADX + more extreme Z-score + higher RVOL
    for c in candidates:
        c["_score"] = (
            min(c["adx"], 50) / 50 * 0.3 +
            min(abs(c["zscore_50"]), 3) / 3 * 0.3 +
            min(c["rvol"], 3) / 3 * 0.2 +
            (1 if c["rsi"] < 30 or c["rsi"] > 70 else 0) * 0.2
        )
    candidates.sort(key=lambda x: x["_score"], reverse=True)

    # Strip internal score
    for c in candidates:
        del c["_score"]

    return candidates


if __name__ == "__main__":
    import sys
    import json
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-03-14"
    results = run_screener(date)
    print(f"Screener found {len(results)} candidates for {date}")
    print(json.dumps(results[:20], indent=2))
