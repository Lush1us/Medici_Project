"""
Capo Office Tool — Macro Scanner

Scans broad market instruments for regime signals.
Isolated to the Capo's office — sub-departments don't have access to this.
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "eod")


def _load(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def scan_macro(as_of_date: str) -> dict:
    """Scan macro instruments and return structured data for the Capo."""
    result = {}

    for ticker in ["SPY", "QQQ", "HYG", "TLT", "VIX"]:
        df = _load(ticker)
        if df is None:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 50:
            continue

        close = df["Close"]
        c = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change_1d = round((c / prev - 1) * 100, 2)

        sma_20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
        sma_50 = round(float(close.rolling(50).mean().iloc[-1]), 2)
        sma_200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(df) >= 200 else None

        change_5d = round((c / float(close.iloc[-5]) - 1) * 100, 2) if len(df) >= 5 else None
        change_20d = round((c / float(close.iloc[-20]) - 1) * 100, 2) if len(df) >= 20 else None

        result[ticker] = {
            "close": round(c, 2),
            "change_1d": change_1d,
            "change_5d": change_5d,
            "change_20d": change_20d,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "above_sma_50": c > sma_50,
            "above_sma_200": c > sma_200 if sma_200 else None,
        }

    # Breadth: % of S&P 500 above 200 SMA
    tickers_file = os.path.join(PROJECT_ROOT, "data", "sp500_tickers.txt")
    if os.path.exists(tickers_file):
        with open(tickers_file) as f:
            sp500 = [t.strip() for t in f if t.strip()]

        above_200 = 0
        total = 0
        for t in sp500:
            df = _load(t)
            if df is None:
                continue
            df = df.loc[:as_of_date]
            if len(df) < 200:
                continue
            total += 1
            if float(df["Close"].iloc[-1]) > float(df["Close"].rolling(200).mean().iloc[-1]):
                above_200 += 1

        result["breadth"] = {
            "pct_above_200sma": round(above_200 / total * 100, 1) if total > 0 else None,
            "total_stocks": total,
        }

    # Credit spread proxy: HYG / TLT ratio
    hyg = _load("HYG")
    tlt = _load("TLT")
    if hyg is not None and tlt is not None:
        hyg = hyg.loc[:as_of_date]
        tlt = tlt.loc[:as_of_date]
        if len(hyg) >= 20 and len(tlt) >= 20:
            ratio = hyg["Close"] / tlt["Close"]
            current = float(ratio.iloc[-1])
            avg_20 = float(ratio.rolling(20).mean().iloc[-1])
            result["credit_spread_proxy"] = {
                "hyg_tlt_ratio": round(current, 4),
                "ratio_vs_20d_avg": round((current / avg_20 - 1) * 100, 2),
                "tightening": current > avg_20,
            }

    return result
