"""
Department of Technical Analysis — Daily Report

Produces a TechnicalReport for each ticker using indicator tools.
No access to macro data — that's the Capo's domain.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from schemas import TechnicalReport, Conviction

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "eod")


def _load(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def compute_technical(ticker: str, as_of_date: str) -> dict | None:
    """Compute core technical indicators for a ticker. Pure Python, no LLM."""
    df = _load(ticker)
    if df is None:
        return None
    df = df.loc[:as_of_date]
    if len(df) < 200:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    c = float(close.iloc[-1])

    # Trend
    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1])
    ema_12 = float(close.ewm(span=12).mean().iloc[-1])
    ema_26 = float(close.ewm(span=26).mean().iloc[-1])

    # MACD
    macd = ema_12 - ema_26
    signal = float(pd.Series([macd]).ewm(span=9).mean().iloc[-1])
    macd_hist = macd - signal

    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = float(100 - (100 / (1 + rs)).iloc[-1])

    # ADX
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0).clip(lower=0)
    minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(14).mean() / tr.rolling(14).mean()
    minus_di = 100 * minus_dm.rolling(14).mean() / tr.rolling(14).mean()
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = float(dx.rolling(14).mean().iloc[-1])

    # Volume
    avg_vol = float(volume.tail(20).mean())
    rvol = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    # Z-scores
    zscore_50 = (c - sma_50) / float(close.rolling(50).std().iloc[-1]) if float(close.rolling(50).std().iloc[-1]) > 0 else 0
    zscore_200 = (c - sma_200) / float(close.rolling(200).std().iloc[-1]) if float(close.rolling(200).std().iloc[-1]) > 0 else 0

    # Bollinger Bands
    bb_mid = sma_50
    bb_std = float(close.rolling(20).std().iloc[-1])
    bb_upper = float(close.rolling(20).mean().iloc[-1]) + 2 * bb_std
    bb_lower = float(close.rolling(20).mean().iloc[-1]) - 2 * bb_std
    pct_b = (c - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # ROC
    roc_12 = (c / float(close.iloc[-12]) - 1) * 100 if len(close) >= 12 else 0

    # Key levels
    high_20 = float(high.tail(20).max())
    low_20 = float(low.tail(20).min())

    # Determine trend
    if c > sma_50 > sma_200:
        trend = "Bullish"
    elif c < sma_50 < sma_200:
        trend = "Bearish"
    else:
        trend = "Neutral"

    # Active signals
    signals = []
    if macd_hist > 0 and float(close.diff().iloc[-1]) > 0:
        signals.append("MACD bullish")
    elif macd_hist < 0:
        signals.append("MACD bearish")
    if rsi > 70:
        signals.append("RSI overbought")
    elif rsi < 30:
        signals.append("RSI oversold")
    if rvol > 2.0:
        signals.append(f"RVOL surge {rvol:.1f}x")
    if abs(zscore_50) > 2.0:
        signals.append(f"Z-score extreme {zscore_50:.1f}")
    if adx > 25:
        signals.append(f"Strong trend ADX={adx:.0f}")
    if c > high_20 * 0.99:
        signals.append("Near 20d high")
    if c < low_20 * 1.01:
        signals.append("Near 20d low")

    # Momentum score: -10 to 10
    momentum = 0.0
    momentum += min(max((rsi - 50) / 5, -3), 3)         # RSI contribution
    momentum += min(max(macd_hist * 2, -2), 2)           # MACD contribution
    momentum += min(max(roc_12 / 3, -3), 3)              # ROC contribution
    momentum += min(max((adx - 20) / 10, -2), 2) * (1 if c > sma_50 else -1)  # ADX directional
    momentum = round(max(min(momentum, 10), -10), 1)

    # Conviction
    signal_count = len(signals)
    if signal_count >= 3 and abs(momentum) >= 5:
        conviction = "high"
    elif signal_count >= 2 or abs(momentum) >= 3:
        conviction = "medium"
    else:
        conviction = "low"

    return {
        "ticker": ticker,
        "date": as_of_date,
        "trend": trend,
        "momentum": momentum,
        "key_levels": {
            "support": round(low_20, 2),
            "resistance": round(high_20, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
        },
        "signals": signals,
        "conviction": conviction,
        "summary": f"{ticker} {trend} trend, momentum {momentum:+.1f}. RSI {rsi:.0f}, ADX {adx:.0f}, RVOL {rvol:.1f}x. {'; '.join(signals[:3]) if signals else 'No active signals'}.",
        # Raw values for other departments
        "_raw": {
            "close": round(c, 2),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "macd_hist": round(macd_hist, 4),
            "rvol": round(rvol, 2),
            "zscore_50": round(zscore_50, 2),
            "zscore_200": round(zscore_200, 2),
            "atr": round(atr, 2),
            "pct_b": round(pct_b, 2),
            "roc_12": round(roc_12, 2),
        },
    }


def run_daily(ticker: str, as_of_date: str) -> dict | None:
    """Produce a daily technical report for one ticker."""
    return compute_technical(ticker, as_of_date)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    date = sys.argv[2] if len(sys.argv) > 2 else "2025-03-10"
    result = run_daily(ticker, date)
    if result:
        # Don't print _raw in standalone mode
        display = {k: v for k, v in result.items() if k != "_raw"}
        print(json.dumps(display, indent=2))
