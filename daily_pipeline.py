"""
Medici — Daily Pipeline (Executive Briefing)

Fast daily execution on the Watchlist.
Steps: Anomaly Scan → Full Indicators → Solo Votes → Filter HOLDs → Capo Allocation

Usage:
    python daily_pipeline.py [date] [watchlist_date] [capo]
    python daily_pipeline.py 2025-03-10 2025-03-07 Lorenzo
"""

import json
import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import requests

from indicators import compute_all
from registry import get_registry_for_scoring
from context import build_context, load_vix
from portfolio import Portfolio
from anomaly_scanner import scan_anomalies
from pipeline import (
    load_ticker, load_optional, DATA_DIR
)

CONSIGLIO_DIR = os.path.join(os.path.dirname(__file__), "Consiglio")
WATCHLIST_DIR = os.path.join(os.path.dirname(__file__), "data", "watchlists")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "daily_runs")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

PERSONAS = ["Cosimo", "Lorenzo", "Giuliano", "Giovanni", "Caterina", "Piero"]
STOCK_PICKERS = ["Cosimo", "Lorenzo", "Giuliano", "Caterina", "Piero"]

MAX_ANOMALIES = 5  # max outsiders injected per day


# =============================================================================
# HELPERS
# =============================================================================

def _load_persona_condensed(name: str) -> str:
    path = os.path.join(CONSIGLIO_DIR, name, "brain.md")
    with open(path) as f:
        text = f.read()
    lines = text.split("\n")
    keep = []
    in_section = False
    for line in lines:
        if line.startswith("## Identity") or line.startswith("## Trading Style") or line.startswith("## Indicator Preferences"):
            in_section = True
            keep.append(line)
        elif line.startswith("## ") and in_section:
            in_section = False
        elif in_section:
            keep.append(line)
    return "\n".join(keep)


def _qwen_call(system: str, user_msg: str, label: str = "") -> dict | None:
    try:
        resp = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"      {label} FAILED: {e}")
        return None


def _parse_wait_time(error_text: str, default: int = 60) -> int:
    import re
    patterns = [r'(\d+)\s*(?:seconds?|s)\b', r'retry.after.*?(\d+)', r'wait.*?(\d+)']
    for p in patterns:
        m = re.search(p, error_text, re.IGNORECASE)
        if m:
            return int(m.group(1)) + 5
    return default


def truncate_to_date(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    return df.loc[:as_of_date]


def load_watchlist(watchlist_date: str) -> dict:
    path = os.path.join(WATCHLIST_DIR, f"watchlist_{watchlist_date}.json")
    if not os.path.exists(path):
        return {"watchlist": [], "macro_regime": None}
    with open(path) as f:
        return json.load(f)


# =============================================================================
# STEP 1: ANOMALY SCAN + QUICK VOTE
# =============================================================================

ANOMALY_VOTE_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

An outsider stock NOT on the watchlist just had an abnormal move. Should we add it to today's analysis?

Respond in valid JSON only. No markdown. Format:
{{
  "vote": "yes" | "no",
  "reason": "one sentence"
}}"""


def vote_anomaly(persona_name: str, anomaly: dict) -> dict | None:
    persona_text = _load_persona_condensed(persona_name)
    system = ANOMALY_VOTE_SYSTEM.format(name=persona_name, persona=persona_text)
    user_msg = f"""{anomaly['ticker']}: {anomaly['daily_change_pct']:+.1f}% today | Triggers: {', '.join(anomaly['triggers'])} | Close=${anomaly['close']} RVOL={anomaly['rvol']}

Should we analyze this outsider today?"""
    return _qwen_call(system, user_msg, f"{persona_name} anomaly({anomaly['ticker']})")


# =============================================================================
# STEP 2: SOLO VOTE (BUY / SELL / HOLD)
# =============================================================================

WATCHLIST_VOTE_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

This ticker is on the watchlist. You are looking for reasons to BUY it today based on your strategy. Only PASS if the setup clearly contradicts your edge or the signals are flat/neutral.

{macro_context}

Respond in valid JSON only. No markdown. Format:
{{
  "vote": "BUY" | "PASS",
  "reasoning": "1 sentence"
}}"""

POSITION_VOTE_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

This ticker is currently HELD in the portfolio. Should we add more, sell, or hold?

{macro_context}

Respond in valid JSON only. No markdown. Format:
{{
  "vote": "BUY" | "SELL" | "HOLD",
  "reasoning": "2 sentences max"
}}"""


def run_watchlist_vote(persona_name: str, payload: dict,
                       macro_regime: dict | None) -> dict | None:
    """Vote BUY or PASS on a watchlist ticker not in portfolio."""
    persona_text = _load_persona_condensed(persona_name)
    macro_ctx = ""
    if macro_regime:
        macro_ctx = f"Giovanni's macro read: {macro_regime.get('regime', '?')} — {macro_regime.get('reasoning', '')}"

    system = WATCHLIST_VOTE_SYSTEM.format(name=persona_name, persona=persona_text, macro_context=macro_ctx)
    ind_summary = {k: v for k, v in payload["indicators"].items()}
    user_msg = f"""{payload['ticker']} on {payload['date']}:
Context: {json.dumps(payload['context'])}
Indicators: {json.dumps(ind_summary)}

Vote: BUY or PASS."""
    return _qwen_call(system, user_msg, f"{persona_name} wl_vote({payload['ticker']})")


def run_position_vote(persona_name: str, payload: dict,
                      macro_regime: dict | None) -> dict | None:
    """Vote BUY/SELL/HOLD on a ticker already in the portfolio."""
    persona_text = _load_persona_condensed(persona_name)
    macro_ctx = ""
    if macro_regime:
        macro_ctx = f"Giovanni's macro read: {macro_regime.get('regime', '?')} — {macro_regime.get('reasoning', '')}"

    system = POSITION_VOTE_SYSTEM.format(name=persona_name, persona=persona_text, macro_context=macro_ctx)
    ind_summary = {k: v for k, v in payload["indicators"].items()}
    user_msg = f"""{payload['ticker']} on {payload['date']}:
Context: {json.dumps(payload['context'])}
Indicators: {json.dumps(ind_summary)}

Vote: BUY, SELL, or HOLD."""
    return _qwen_call(system, user_msg, f"{persona_name} pos_vote({payload['ticker']})")


# =============================================================================
# GIOVANNI DAILY MACRO UPDATE
# =============================================================================

GIOVANNI_DAILY_SYSTEM = """You are Giovanni, a macro specialist. Key traits:
{persona}

Update your macro regime call for today.

Respond in valid JSON only. No markdown. Format:
{{
  "regime": "risk-on" | "risk-off" | "transitional",
  "confidence": 1-10,
  "reasoning": "2-3 sentences"
}}"""


def run_giovanni_daily(as_of_date: str) -> dict | None:
    persona_text = _load_persona_condensed("Giovanni")
    system = GIOVANNI_DAILY_SYSTEM.format(persona=persona_text)

    macro = {}
    for ticker in ["SPY", "HYG", "TLT", "VIX"]:
        df = load_optional(ticker) if ticker != "SPY" else load_ticker("SPY")
        if df is None:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 20:
            continue
        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        change = round((close / prev - 1) * 100, 2)
        sma_50 = round(float(df["Close"].rolling(50).mean().iloc[-1]), 2) if len(df) >= 50 else None
        macro[ticker] = {"close": round(close, 2), "daily_change": change, "sma_50": sma_50}

    user_msg = f"Date: {as_of_date}\nMacro: {json.dumps(macro)}"
    return _qwen_call(system, user_msg, "Giovanni daily")


# =============================================================================
# CAPO DECISION (Claude CLI)
# =============================================================================

CAPO_SYSTEM = """You are the Capo of the Consiglio — the final decision maker.

## Your Role
You receive today's actionable list: tickers with vote tallies and reasoning from your advisors. Allocate the portfolio.

## Investment Mandate
- Deploy at least 80% of the portfolio. Cash sitting idle is failure.
- Invested capital: 75-90% of total portfolio value at all times.
- If invested % drops below 75%, you MUST buy.
- If invested % rises above 90%, you MUST trim.
- Max 8 concurrent positions.

## Rules
- Portfolio starts at $1000 with fractional shares.
- Position sizes in dollar amounts.
- Max single trade: 5% of total portfolio value. Spread risk across positions.
- Actively manage existing positions (hold, add, trim, close).
- Weight your decisions by vote consensus — unanimous BUY is stronger than 3-2.

Respond in valid JSON only. No markdown. Format:
{{
  "date": "YYYY-MM-DD",
  "invested_pct": N,
  "actions": [
    {{"ticker": "SYMBOL", "action": "buy" | "sell" | "sell_all" | "hold", "amount_usd": N, "reason": "brief"}}
  ],
  "rationale": "2-3 sentence assessment",
  "risk_notes": "portfolio-level concerns"
}}"""


def run_capo(actionable_summary: str, portfolio_state: str,
             as_of_date: str) -> dict | None:
    user_msg = f"""Date: {as_of_date}

## Today's Actionable List
{actionable_summary}

## Current Portfolio
{portfolio_state}

Allocate the portfolio."""

    full_prompt = f"{CAPO_SYSTEM}\n\n{user_msg}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["claude", "-p", "--output-format", "json", "--model", "claude-sonnet-4-6", "--effort", "medium"],
                input=full_prompt,
                capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                if any(k in stderr.lower() for k in ["rate", "429", "limit", "overloaded"]):
                    wait = _parse_wait_time(stderr, default=60)
                    print(f"      Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"      Claude error: {stderr[:200]}")
                return None

            output = json.loads(result.stdout)
            raw = output.get("result", "")
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            cleaned = cleaned.strip()

            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(cleaned[start:end])

            print(f"      No valid JSON in Claude response")
            return None

        except subprocess.TimeoutExpired:
            print(f"      Claude timed out (attempt {attempt+1}/{max_retries})")
            continue
        except json.JSONDecodeError as e:
            print(f"      JSON parse error: {e}")
            return None
        except Exception as e:
            stderr = str(e)
            if any(k in stderr.lower() for k in ["rate", "429", "limit"]):
                wait = _parse_wait_time(stderr, default=60)
                print(f"      Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"      Claude error: {e}")
            return None

    print(f"      Max retries exceeded")
    return None


# =============================================================================
# TRADE EXECUTION
# =============================================================================

def get_next_day_open(ticker: str, as_of_date: str) -> float | None:
    df = load_ticker(ticker)
    future = df.loc[as_of_date:]
    if len(future) < 2:
        return None
    return float(future.iloc[1]["Open"])


MAX_TRADE_PCT = 0.05  # hard cap: 5% of portfolio per trade

def execute_actions(portfolio: Portfolio, actions: list, as_of_date: str,
                    current_prices: dict = None):
    # Compute portfolio value for position cap
    prices = current_prices or get_current_prices(
        list(portfolio.positions.keys()), as_of_date)
    total_value = portfolio.get_total_value(prices)
    max_trade_usd = total_value * MAX_TRADE_PCT

    for action in actions:
        ticker = action.get("ticker")
        act = action.get("action")
        amount_usd = action.get("amount_usd", 0)

        if act == "hold" or not ticker:
            continue

        next_open = get_next_day_open(ticker, as_of_date)
        if next_open is None:
            print(f"      No next-day data for {ticker}, skipping")
            continue

        df = load_ticker(ticker)
        future = df.loc[as_of_date:]
        if len(future) < 2:
            continue
        exec_date = future.index[1].strftime("%Y-%m-%d")

        if act == "buy":
            # Enforce 5% cap
            if amount_usd > max_trade_usd:
                print(f"      CAP: {ticker} buy ${amount_usd:.2f} → ${max_trade_usd:.2f} (5% limit)")
                amount_usd = max_trade_usd
            shares = amount_usd / next_open if next_open > 0 else 0
            if shares > 0:
                ok = portfolio.buy(ticker, shares, next_open, exec_date)
                if ok:
                    print(f"      EXEC: BUY {shares:.4f} {ticker} @ ${next_open:.2f} (${amount_usd:.2f})")
        elif act == "sell":
            shares = amount_usd / next_open if next_open > 0 else 0
            if shares > 0:
                ok = portfolio.sell(ticker, shares, next_open, exec_date)
                if ok:
                    print(f"      EXEC: SELL {shares:.4f} {ticker} @ ${next_open:.2f}")
        elif act == "sell_all":
            ok = portfolio.sell_all(ticker, next_open, exec_date)
            if ok:
                print(f"      EXEC: SELL ALL {ticker} @ ${next_open:.2f}")


def get_current_prices(tickers: list, as_of_date: str) -> dict:
    prices = {}
    for t in tickers:
        try:
            df = load_ticker(t)
            trunc = df.loc[:as_of_date]
            if len(trunc) > 0:
                prices[t] = float(trunc.iloc[-1]["Close"])
        except Exception:
            pass
    return prices


# =============================================================================
# MAIN DAILY PIPELINE
# =============================================================================

def run_daily(as_of_date: str, watchlist_date: str,
              capo: str = "Lorenzo", portfolio: Portfolio = None,
              run_dir: str = None) -> dict:
    """Run one day of the daily pipeline. Returns day result dict."""

    if run_dir is None:
        run_dir = os.path.join(OUTPUT_DIR, f"daily_{watchlist_date}_{capo}")
    os.makedirs(run_dir, exist_ok=True)

    if portfolio is None:
        portfolio = Portfolio(starting_cash=1000.0, log_dir=run_dir)

    print(f"\n  {'='*60}")
    print(f"  DAILY PIPELINE — {as_of_date}")
    print(f"  {'='*60}")

    # Load watchlist
    wl_data = load_watchlist(watchlist_date)
    watchlist_tickers = [w["ticker"] for w in wl_data.get("watchlist", [])]
    macro_regime = wl_data.get("macro_regime")

    print(f"    Watchlist: {len(watchlist_tickers)} tickers from {watchlist_date}")

    # =========================================================================
    # STEP 0: ANOMALY SCAN
    # =========================================================================
    print(f"    [0] Anomaly scan...")
    t0 = time.time()
    anomalies = scan_anomalies(as_of_date, set(watchlist_tickers))
    top_anomalies = anomalies[:MAX_ANOMALIES]
    print(f"        {len(anomalies)} anomalies found, evaluating top {len(top_anomalies)}")

    injected = []
    for anom in top_anomalies:
        yes_votes = 0
        for persona in STOCK_PICKERS:
            v = vote_anomaly(persona, anom)
            if v and v.get("vote") == "yes":
                yes_votes += 1
        if yes_votes >= 3:  # majority
            injected.append(anom["ticker"])
            print(f"        {anom['ticker']}: INJECTED ({yes_votes}/5 yes) — {', '.join(anom['triggers'])}")
        else:
            print(f"        {anom['ticker']}: skipped ({yes_votes}/5 yes)")

    # Combined ticker list for today
    todays_tickers = watchlist_tickers + injected
    print(f"        Today's universe: {len(todays_tickers)} tickers ({len(injected)} injected)")
    print(f"        ({time.time()-t0:.0f}s)")

    if not todays_tickers:
        print(f"        Empty universe, nothing to do")
        prices = get_current_prices(list(portfolio.positions.keys()), as_of_date)
        snap = portfolio.snapshot(as_of_date, prices)
        return {"date": as_of_date, "portfolio_snapshot": snap}

    # =========================================================================
    # STEP 0.5: GIOVANNI DAILY MACRO UPDATE
    # =========================================================================
    print(f"    [0.5] Giovanni macro update...")
    t0 = time.time()
    daily_macro = run_giovanni_daily(as_of_date)
    if daily_macro:
        macro_regime = daily_macro  # override weekend regime with fresh read
        print(f"          Regime: {daily_macro.get('regime', '?')} — {daily_macro.get('reasoning', '')[:80]}")
    print(f"          ({time.time()-t0:.0f}s)")

    # =========================================================================
    # STEP 1: FULL INDICATORS (only on today's tickers)
    # =========================================================================
    print(f"    [1] Computing indicators...")
    try:
        full_spy = load_ticker("SPY")
    except FileNotFoundError:
        print(f"        FATAL: SPY data missing, cannot continue")
        prices = get_current_prices(list(portfolio.positions.keys()), as_of_date)
        snap = portfolio.snapshot(as_of_date, prices)
        return {"date": as_of_date, "portfolio_snapshot": snap, "error": "SPY data missing"}
    full_vix = load_vix()
    full_hyg = load_optional("HYG")
    full_tlt = load_optional("TLT")

    payloads = {}
    for ticker in todays_tickers:
        try:
            t_df = load_ticker(ticker)
            t_df = truncate_to_date(t_df, as_of_date)
            if len(t_df) < 200:
                print(f"        {ticker}: insufficient data, skipping")
                continue

            spy_trunc = truncate_to_date(full_spy, as_of_date)
            vix_trunc = truncate_to_date(full_vix, as_of_date) if full_vix is not None else None
            hyg_trunc = truncate_to_date(full_hyg, as_of_date) if full_hyg is not None else None
            tlt_trunc = truncate_to_date(full_tlt, as_of_date) if full_tlt is not None else None

            t0 = time.time()
            all_ind = compute_all(t_df, spy_df=spy_trunc, hyg_df=hyg_trunc, tlt_df=tlt_trunc)
            ctx = build_context(ticker, t_df, vix_trunc)

            # Fixed indicator set — normalized signals agents can actually interpret
            KEEP_INDICATORS = {
                # Trend / Momentum
                "ADX_14", "RSI_14", "MACD_12_26_9", "CCI_20", "ROC_12",
                "STOCH_14_3_3", "ELDER_IMPULSE", "MOM_ACCEL_5",
                # Mean reversion
                "ZSCORE_50", "ZSCORE_200", "BBANDS_20_2",
                # Volume
                "RVOL_20", "CMF_20", "MFI_14", "UP_DOWN_VOL_RATIO_20",
                # Volatility
                "NATR_14", "VOL_RATIO_10_60", "HURST_100",
                # Microstructure
                "AMIHUD_20", "CLV", "BAR_RANGE_RATIO",
                # Cross-asset
                "BETA_SPY_60", "CORR_SPY_20",
            }
            valid = {k: v for k, v in all_ind.items() if v is not None and k in KEEP_INDICATORS}

            payload = {
                "ticker": ticker,
                "date": as_of_date,
                "context": ctx,
                "indicators": valid,
                "scoring_summary": {"total_evaluated": len(all_ind), "kept": len(valid)},
            }
            payloads[ticker] = payload
            print(f"        {ticker}: {len(valid)} indicators ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"        {ticker}: ERROR — {e}")

    # =========================================================================
    # STEP 2: VOTES (BUY/PASS for watchlist, BUY/SELL/HOLD for held)
    # =========================================================================
    print(f"    [2] Voting...")

    held_tickers = set(portfolio.positions.keys())
    all_votes = {}  # {ticker: {persona: vote_result}}
    BET_PER_VOTE = 10.0  # $10 per BUY vote

    for ticker, payload in payloads.items():
        all_votes[ticker] = {}
        is_held = ticker in held_tickers

        # Stock pickers vote on their own signals (no macro context)
        for persona in STOCK_PICKERS:
            if is_held:
                v = run_position_vote(persona, payload, None)
            else:
                v = run_watchlist_vote(persona, payload, None)
            all_votes[ticker][persona] = v

        # Giovanni votes through macro lens (he IS the macro context)
        gio_vote = _giovanni_ticker_vote(ticker, payload, macro_regime, is_held)
        all_votes[ticker]["Giovanni"] = gio_vote

        # Tally
        votes = all_votes[ticker]
        buy = sum(1 for v in votes.values() if v and v.get("vote") == "BUY")
        if is_held:
            sell = sum(1 for v in votes.values() if v and v.get("vote") == "SELL")
            hold = sum(1 for v in votes.values() if v and v.get("vote") == "HOLD")
            print(f"        {ticker:6s} [HELD]: BUY={buy} SELL={sell} HOLD={hold}")
        else:
            pas = sum(1 for v in votes.values() if v and v.get("vote") == "PASS")
            print(f"        {ticker:6s}: BUY={buy} PASS={pas}")

    # =========================================================================
    # STEP 3: MECHANICAL EXECUTION ($10 per BUY vote, sell on majority SELL)
    # =========================================================================
    print(f"    [3] Executing...")

    actions = []
    for ticker, votes in all_votes.items():
        buy_count = sum(1 for v in votes.values() if v and v.get("vote") == "BUY")
        is_held = ticker in held_tickers

        if is_held:
            sell_count = sum(1 for v in votes.values() if v and v.get("vote") == "SELL")
            if sell_count >= 4:  # strong majority says sell
                actions.append({"ticker": ticker, "action": "sell_all", "amount_usd": 0,
                                "reason": f"{sell_count}/6 SELL"})
            elif buy_count >= 1:  # add to position
                amount = buy_count * BET_PER_VOTE
                actions.append({"ticker": ticker, "action": "buy", "amount_usd": amount,
                                "reason": f"{buy_count}/6 BUY → ${amount:.0f}"})
        else:
            if buy_count >= 1:  # at least one agent says buy
                amount = buy_count * BET_PER_VOTE
                actions.append({"ticker": ticker, "action": "buy", "amount_usd": amount,
                                "reason": f"{buy_count}/6 BUY → ${amount:.0f}"})

    if actions:
        print(f"        {len(actions)} actions:")
        for a in actions:
            print(f"          {a['action'].upper():8s} {a['ticker']:6s} ${a['amount_usd']:.0f} — {a['reason']}")
        execute_actions(portfolio, actions, as_of_date)
    else:
        print(f"        No actions today")

    # Snapshot
    all_held = list(portfolio.positions.keys())
    prices_now = get_current_prices(all_held, as_of_date)
    snap = portfolio.snapshot(as_of_date, prices_now)
    print(f"    [>>] Portfolio: ${snap['total_value']:.2f} ({snap['total_return_pct']:+.2f}%)")
    portfolio.save()

    # Save day
    day_result = {
        "date": as_of_date,
        "watchlist_tickers": watchlist_tickers,
        "injected_anomalies": injected,
        "macro_regime": macro_regime,
        "votes": all_votes,
        "actions": actions,
        "portfolio_snapshot": snap,
    }
    return day_result


def _giovanni_ticker_vote(ticker: str, payload: dict, macro_regime: dict | None,
                          is_held: bool = False) -> dict | None:
    """Giovanni votes on a ticker purely through macro lens."""
    persona_text = _load_persona_condensed("Giovanni")
    regime = macro_regime.get("regime", "transitional") if macro_regime else "transitional"

    if is_held:
        vote_options = '"BUY" | "SELL" | "HOLD"'
        vote_prompt = "Vote: BUY, SELL, or HOLD."
    else:
        vote_options = '"BUY" | "PASS"'
        vote_prompt = "Vote: BUY or PASS."

    system = f"""You are Giovanni, a macro specialist. Traits:\n{persona_text}\n\nYour current macro read: {regime}. Vote on this ticker through your macro lens only.\n\nRespond in valid JSON only. No markdown. Format:\n{{"vote": {vote_options}, "reasoning": "2 sentences max"}}"""

    ctx = payload.get("context", {})
    user_msg = f"""{ticker}: Close=${ctx.get('close','?')} Change={ctx.get('change_pct','?')}% Regime={ctx.get('regime','?')}
Your macro read is {regime}. {vote_prompt}"""

    return _qwen_call(system, user_msg, f"Giovanni vote({ticker})")


# =============================================================================
# BACKTEST HARNESS: Run daily pipeline across multiple days
# =============================================================================

def run_backtest(start_date: str, end_date: str, watchlist_date: str,
                 capo: str = "Lorenzo", starting_cash: float = 1000.0):
    """Run the daily pipeline across a date range using one watchlist."""
    run_id = f"daily_{watchlist_date}_{start_date}_{end_date}_{capo}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MEDICI DAILY BACKTEST")
    print(f"  Watchlist from: {watchlist_date}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Capo: {capo}")
    print(f"  Cash: ${starting_cash:.2f}")
    print(f"{'='*70}")

    spy_df = load_ticker("SPY")
    mask = (spy_df.index >= start_date) & (spy_df.index <= end_date)
    trading_days = [d.strftime("%Y-%m-%d") for d in spy_df.index[mask]]
    print(f"\n  {len(trading_days)} trading days\n")

    portfolio = Portfolio(starting_cash=starting_cash, log_dir=run_dir)

    WIPEOUT_THRESHOLD = 0.10  # stop if portfolio drops to 10% of starting cash

    for day_num, date in enumerate(trading_days, 1):
        print(f"\n  DAY {day_num}/{len(trading_days)}")
        result = run_daily(date, watchlist_date, capo=capo,
                          portfolio=portfolio, run_dir=run_dir)

        # Save day file
        day_path = os.path.join(run_dir, f"day_{day_num:03d}_{date}.json")
        with open(day_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Circuit breaker: stop if wiped out
        snap = result.get("portfolio_snapshot", {})
        total = snap.get("total_value", portfolio.cash)
        if total < starting_cash * WIPEOUT_THRESHOLD:
            print(f"\n  CIRCUIT BREAKER: Portfolio at ${total:.2f} (<{WIPEOUT_THRESHOLD*100:.0f}% of starting). Stopping.")
            break

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"BACKTEST COMPLETE")
    print(f"{'='*70}")
    if portfolio.daily_snapshots:
        final = portfolio.daily_snapshots[-1]
        print(f"  Final Value:  ${final['total_value']:.2f}")
        print(f"  Total Return: {final['total_return_pct']:+.2f}%")
        print(f"  Total Trades: {len(portfolio.trade_log)}")
        values = [s["total_value"] for s in portfolio.daily_snapshots]
        peak = max(values)
        trough = min(values)
        max_dd = (trough - peak) / peak * 100 if peak > 0 else 0
        print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"\n  Results: {run_dir}")
    portfolio.save()


def run_rolling_backtest(start_date: str, end_date: str,
                         capo: str = "Lorenzo", starting_cash: float = 1000.0,
                         skip_first_weekend: bool = False):
    """Rolling backtest: weekend → 5 daily → weekend → 5 daily → ... until end or wipeout.

    Walks forward through trading days. Every 5 trading days, rebuilds the watchlist
    using the Friday (last trading day of the week) as the weekend pipeline date.

    Args:
        skip_first_weekend: If True, assumes a watchlist already exists for the
            Friday before start_date and jumps straight into daily trading.
    """
    from weekend_pipeline import run_weekend_pipeline

    run_id = f"rolling_{start_date}_{end_date}_{capo}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MEDICI ROLLING BACKTEST")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Capo: {capo}")
    print(f"  Cash: ${starting_cash:.2f}")
    print(f"  Skip first weekend: {skip_first_weekend}")
    print(f"{'='*70}")

    spy_df = load_ticker("SPY")
    mask = (spy_df.index >= start_date) & (spy_df.index <= end_date)
    all_trading_days = [d.strftime("%Y-%m-%d") for d in spy_df.index[mask]]
    print(f"\n  {len(all_trading_days)} total trading days\n")

    if not all_trading_days:
        print("  No trading days in range.")
        return

    portfolio = Portfolio(starting_cash=starting_cash, log_dir=run_dir)
    WIPEOUT_THRESHOLD = 0.10

    # Find the Friday before start_date for the first weekend pipeline
    pre_start = spy_df.index[spy_df.index < start_date]
    if len(pre_start) == 0:
        print("  No data before start_date for initial weekend pipeline.")
        return
    first_weekend_date = pre_start[-1].strftime("%Y-%m-%d")  # last trading day before start

    # Split trading days into weeks of 5
    weeks = []
    for i in range(0, len(all_trading_days), 5):
        weeks.append(all_trading_days[i:i+5])

    day_num = 0
    wiped = False

    for week_idx, week_days in enumerate(weeks):
        # Determine weekend date for this week's watchlist
        if week_idx == 0:
            weekend_date = first_weekend_date
        else:
            # Use the last trading day of the previous week
            prev_week = weeks[week_idx - 1]
            weekend_date = prev_week[-1]

        print(f"\n  {'#'*70}")
        print(f"  WEEK {week_idx + 1}: watchlist from {weekend_date}, trading {week_days[0]} → {week_days[-1]}")
        print(f"  {'#'*70}")

        # Weekend pipeline (skip if watchlist exists or skip_first_weekend on week 1)
        wl_path = os.path.join(WATCHLIST_DIR, f"watchlist_{weekend_date}.json")
        if skip_first_weekend and week_idx == 0 and os.path.exists(wl_path):
            with open(wl_path) as f:
                wl_data = json.load(f)
            wl_count = len(wl_data.get("watchlist", []))
            print(f"  Using existing watchlist: {wl_count} tickers")
        elif os.path.exists(wl_path):
            with open(wl_path) as f:
                wl_data = json.load(f)
            wl_count = len(wl_data.get("watchlist", []))
            print(f"  Watchlist already exists: {wl_count} tickers, skipping rebuild")
        else:
            print(f"  Running weekend pipeline for {weekend_date}...")
            t0 = time.time()
            run_weekend_pipeline(weekend_date)
            print(f"  Weekend pipeline done in {(time.time()-t0)/60:.0f} min")

        # Merge previous watchlist survivors into the new one (additive)
        if week_idx >= 1 and os.path.exists(wl_path):
            # Find previous week's watchlist
            if week_idx == 1:
                prev_wl_date = first_weekend_date
            else:
                prev_wl_date = weeks[week_idx - 2][-1] if week_idx >= 2 else first_weekend_date
            prev_wl_path = os.path.join(WATCHLIST_DIR, f"watchlist_{prev_wl_date}.json")

            if os.path.exists(prev_wl_path):
                with open(prev_wl_path) as f:
                    prev_data = json.load(f)
                with open(wl_path) as f:
                    wl_data = json.load(f)

                new_tickers = {t["ticker"] for t in wl_data.get("watchlist", [])}
                traded = {t["ticker"] for t in portfolio.trade_log}
                held = set(portfolio.positions.keys())

                # Carry forward previous tickers not already in new watchlist
                carried = 0
                stale = 0
                for entry in prev_data.get("watchlist", []):
                    t = entry["ticker"]
                    if t in new_tickers:
                        continue  # already picked fresh

                    # Check staleness: was it on the watchlist 2 weeks ago too?
                    # If so and never traded/held, drop it
                    if week_idx >= 2:
                        two_ago_date = first_weekend_date if week_idx == 2 else weeks[week_idx - 3][-1]
                        two_ago_path = os.path.join(WATCHLIST_DIR, f"watchlist_{two_ago_date}.json")
                        if os.path.exists(two_ago_path):
                            with open(two_ago_path) as f:
                                two_ago = json.load(f)
                            two_ago_tickers = {x["ticker"] for x in two_ago.get("watchlist", [])}
                            if t in two_ago_tickers and t not in traded and t not in held:
                                stale += 1
                                continue  # stale — 2+ weeks, never traded

                    wl_data["watchlist"].append(entry)
                    carried += 1

                with open(wl_path, "w") as f:
                    json.dump(wl_data, f, indent=2, default=str)
                total = len(wl_data["watchlist"])
                print(f"  Merged: +{carried} carried from prev week, {stale} stale dropped → {total} total")

        # Daily loop for this week
        for date in week_days:
            day_num += 1
            print(f"\n  DAY {day_num} ({date})")

            result = run_daily(date, weekend_date, capo=capo,
                              portfolio=portfolio, run_dir=run_dir)

            day_path = os.path.join(run_dir, f"day_{day_num:03d}_{date}.json")
            with open(day_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Circuit breaker
            snap = result.get("portfolio_snapshot", {})
            total = snap.get("total_value", portfolio.cash)
            if total < starting_cash * WIPEOUT_THRESHOLD:
                print(f"\n  CIRCUIT BREAKER: Portfolio at ${total:.2f}. Stopping.")
                wiped = True
                break

        if wiped:
            break

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"ROLLING BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Days traded: {day_num}")
    print(f"  Weeks: {week_idx + 1}")
    if portfolio.daily_snapshots:
        final = portfolio.daily_snapshots[-1]
        print(f"  Final Value:  ${final['total_value']:.2f}")
        print(f"  Total Return: {final['total_return_pct']:+.2f}%")
        print(f"  Total Trades: {len(portfolio.trade_log)}")
        values = [s["total_value"] for s in portfolio.daily_snapshots]
        peak = max(values)
        trough = min(values)
        max_dd = (trough - peak) / peak * 100 if peak > 0 else 0
        print(f"  Max Drawdown: {max_dd:.2f}%")
    if wiped:
        print(f"  STOPPED: Circuit breaker triggered")
    print(f"\n  Results: {run_dir}")
    portfolio.save()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "rolling":
        # Rolling mode: weekend → 5 daily → weekend → ...
        # python daily_pipeline.py rolling <start> <end> [capo] [cash] [--skip-first-weekend]
        if len(sys.argv) < 4:
            print("Usage: python daily_pipeline.py rolling <start> <end> [capo] [cash] [--skip-first-weekend]")
            sys.exit(1)
        start = sys.argv[2]
        end = sys.argv[3]
        capo = "Lorenzo"
        cash = 1000.0
        skip = False
        for arg in sys.argv[4:]:
            if arg == "--skip-first-weekend":
                skip = True
            elif arg.replace(".", "").isdigit():
                cash = float(arg)
            else:
                capo = arg
        run_rolling_backtest(start, end, capo=capo, starting_cash=cash,
                             skip_first_weekend=skip)
    elif len(sys.argv) >= 4:
        # Backtest mode: start_date end_date watchlist_date [capo]
        start = sys.argv[1]
        end = sys.argv[2]
        wl_date = sys.argv[3]
        capo = sys.argv[4] if len(sys.argv) > 4 else "Lorenzo"
        run_backtest(start, end, wl_date, capo=capo)
    elif len(sys.argv) >= 3:
        # Single day: date watchlist_date [capo]
        date = sys.argv[1]
        wl_date = sys.argv[2]
        capo = sys.argv[3] if len(sys.argv) > 3 else "Lorenzo"
        run_daily(date, wl_date, capo=capo)
    else:
        print("Usage:")
        print("  Rolling:     python daily_pipeline.py rolling <start> <end> [capo] [cash] [--skip-first-weekend]")
        print("  Daily range: python daily_pipeline.py <start> <end> <watchlist_date> [capo]")
        print("  Single day:  python daily_pipeline.py <date> <watchlist_date> [capo]")
