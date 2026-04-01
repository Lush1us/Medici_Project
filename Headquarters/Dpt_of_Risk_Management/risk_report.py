"""
Department of Risk Management — Daily Report

Produces a RiskReport for each ticker. Evaluates position sizing,
stop losses, correlation, and risk flags. Pure Python, no LLM.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "eod")
LEDGER_DIR = "/mnt/iarvis/Library/ledgers/Medici"


def _load(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _load_portfolio() -> dict:
    """Load current portfolio state from ledger."""
    path = os.path.join(LEDGER_DIR, "portfolio.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cash": 1000.0, "positions": {}, "starting_cash": 1000.0}


def compute_risk(ticker: str, as_of_date: str) -> dict | None:
    """Compute risk metrics for a ticker."""
    df = _load(ticker)
    if df is None:
        return None
    df = df.loc[:as_of_date]
    if len(df) < 60:
        return None

    close = df["Close"]
    c = float(close.iloc[-1])

    # Volatility
    returns = np.log(close / close.shift(1)).dropna()
    vol_20 = float(returns.tail(20).std() * np.sqrt(252))
    vol_60 = float(returns.tail(60).std() * np.sqrt(252))

    # ATR for stop loss
    high = df["High"]
    low = df["Low"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(14).mean().iloc[-1])

    # Max drawdown (60d)
    roll_max = close.tail(60).cummax()
    drawdown = (close.tail(60) / roll_max - 1)
    max_dd_60 = float(drawdown.min())

    # Beta to SPY
    spy = _load("SPY")
    beta = None
    corr = None
    if spy is not None:
        spy = spy.loc[:as_of_date]
        if len(spy) >= 60:
            spy_ret = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
            # Align
            common = returns.index.intersection(spy_ret.index)
            if len(common) >= 20:
                t_ret = returns.loc[common].tail(60)
                s_ret = spy_ret.loc[common].tail(60)
                cov = np.cov(t_ret, s_ret)
                if cov[1, 1] > 0:
                    beta = round(float(cov[0, 1] / cov[1, 1]), 2)
                corr = round(float(np.corrcoef(t_ret, s_ret)[0, 1]), 2)

    # Portfolio context
    portfolio = _load_portfolio()
    total_value = portfolio.get("cash", 1000.0)
    for t, pos in portfolio.get("positions", {}).items():
        tdf = _load(t)
        if tdf is not None:
            tdf = tdf.loc[:as_of_date]
            if len(tdf) > 0:
                total_value += pos["shares"] * float(tdf["Close"].iloc[-1])

    current_exposure = 0.0
    pos = portfolio.get("positions", {}).get(ticker)
    if pos:
        current_exposure = pos["shares"] * c / total_value * 100 if total_value > 0 else 0

    # Position sizing: risk 2% of portfolio per trade, stop at 2x ATR
    risk_per_trade = total_value * 0.02
    stop_distance = 2 * atr_14
    max_shares = risk_per_trade / stop_distance if stop_distance > 0 else 0
    max_position_usd = max_shares * c

    # Stop loss: 2x ATR below current price
    stop_loss = round(c - 2 * atr_14, 2)

    # Risk-reward (using 3:1 target)
    risk_reward = 3.0

    # Flags
    flags = []
    if vol_20 > 0.5:
        flags.append("high_volatility")
    if abs(max_dd_60) > 0.15:
        flags.append("deep_recent_drawdown")
    if beta and abs(beta) > 1.5:
        flags.append("high_beta")
    if current_exposure > 10:
        flags.append("concentrated_position")
    if vol_20 > vol_60 * 1.5:
        flags.append("vol_expanding")

    return {
        "ticker": ticker,
        "date": as_of_date,
        "current_exposure_pct": round(current_exposure, 2),
        "max_position_size_usd": round(max_position_usd, 2),
        "stop_loss_price": stop_loss,
        "risk_reward_ratio": risk_reward,
        "correlation_to_portfolio": corr,
        "flags": flags,
        "summary": f"{ticker}: Vol {vol_20:.0%} (20d), Beta {beta}, MaxDD {max_dd_60:.1%} (60d). Stop ${stop_loss:.2f}, MaxSize ${max_position_usd:.0f}. {', '.join(flags) if flags else 'No flags'}.",
        "_raw": {
            "close": round(c, 2),
            "vol_20": round(vol_20, 4),
            "vol_60": round(vol_60, 4),
            "atr_14": round(atr_14, 2),
            "max_dd_60": round(max_dd_60, 4),
            "beta": beta,
            "total_portfolio_value": round(total_value, 2),
        },
    }


def run_daily(ticker: str, as_of_date: str) -> dict | None:
    return compute_risk(ticker, as_of_date)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    date = sys.argv[2] if len(sys.argv) > 2 else "2025-03-10"
    result = run_daily(ticker, date)
    if result:
        display = {k: v for k, v in result.items() if k != "_raw"}
        print(json.dumps(display, indent=2))
