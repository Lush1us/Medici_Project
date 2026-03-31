"""
Medici — Market Context Builder

Builds the regime snapshot that Qwen uses alongside the indicator registry
to make scoring decisions.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")


def build_context(ticker: str, df: pd.DataFrame, vix_df: pd.DataFrame = None) -> dict:
    """
    Build market context dict for a ticker from its OHLCV dataframe.
    Optionally takes VIX dataframe for volatility regime.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest["Close"])
    change_pct = round((close - float(prev["Close"])) / float(prev["Close"]) * 100, 2)

    vol_20 = float(df["Volume"].tail(20).mean())
    vol_ratio = round(float(latest["Volume"]) / vol_20, 2) if vol_20 > 0 else 1.0

    sma_50 = float(df["Close"].rolling(50).mean().iloc[-1])
    sma_200 = float(df["Close"].rolling(200).mean().iloc[-1])

    # Regime label
    regime_parts = []

    # Trend
    if close > sma_50:
        regime_parts.append("Uptrend" if close > sma_200 else "Short-term uptrend below 200-SMA")
    else:
        regime_parts.append("Downtrend" if close < sma_200 else "Pullback above 200-SMA")

    # Price vs MAs
    ma_notes = []
    if close > sma_50:
        ma_notes.append("above 50-SMA")
    else:
        ma_notes.append("below 50-SMA")
    if close > sma_200:
        ma_notes.append("above 200-SMA")
    else:
        ma_notes.append("below 200-SMA")
    regime_parts.append(f"Price {', '.join(ma_notes)}.")

    # VIX
    vix_close = None
    if vix_df is not None and not vix_df.empty:
        vix_close = round(float(vix_df["Close"].iloc[-1]), 1)
        if vix_close > 25:
            regime_parts.append(f"High volatility (VIX {vix_close}).")
        elif vix_close > 18:
            regime_parts.append(f"Elevated volatility (VIX {vix_close}).")
        else:
            regime_parts.append(f"Low volatility (VIX {vix_close}).")

    # Volume
    if vol_ratio > 1.3:
        regime_parts.append(f"Heavy volume ({vol_ratio}x avg).")
    elif vol_ratio < 0.7:
        regime_parts.append(f"Light volume ({vol_ratio}x avg).")

    ctx = {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "close": round(close, 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "change_pct": change_pct,
        "volume_vs_avg": vol_ratio,
        "sma_50": round(sma_50, 2),
        "sma_200": round(sma_200, 2),
        "vix": vix_close,
        "regime": " ".join(regime_parts),
    }

    return ctx


def load_vix() -> pd.DataFrame:
    """Load VIX data from parquet."""
    path = os.path.join(DATA_DIR, "VIX.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()
