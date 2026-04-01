"""
Medici — Backtest Engine v3

4-phase pipeline:
  Phase 1: Python screener scans S&P 500 for candidates (no LLM)
  Phase 2: Agents pitch 0-3 tickers each from screener output (Qwen)
  Phase 3: Deduplicated deep dives — heavy indicators + scoring once per ticker (Qwen)
  Phase 4: Isolated debates per ticker, then executive summary to Capo (Claude)
"""

import json
import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from indicators import compute_all
from registry import get_registry_for_scoring
from context import build_context, load_vix
from portfolio import Portfolio
from screener import run_screener
from pipeline import (
    load_ticker, load_optional, load_universe, load_sectors,
    score_indicators, pack_payload, DATA_DIR
)

CONSIGLIO_DIR = os.path.join(os.path.dirname(__file__), "Consiglio")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "backtests")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

PERSONAS = ["Cosimo", "Lorenzo", "Giuliano", "Giovanni", "Caterina", "Piero"]

MAX_PICKS_PER_AGENT = 3

# =============================================================================
# PHASE 2: Agent Pitch Prompts
# =============================================================================

PITCH_SYSTEM = """You are a trading analyst. Your personality, trading style, and preferences are defined below. Stay in character — your biases are part of the analysis.

## Your Profile
{persona}

## Instructions
You will receive today's screener output: a list of S&P 500 stocks that passed basic regime filters, along with 5 core metrics each (ADX, RSI, Price vs 200-SMA %, RVOL, Z-Score vs 50-SMA).

Your job: select between 0 and {max_picks} tickers that match YOUR trading style. Do not pick tickers just because they're on the list — only pick ones where YOUR edge applies.

You will also receive the current portfolio state. Factor open positions into your picks — don't propose buying what we already hold unless you want to add, and flag positions that should be closed.

Respond in valid JSON only. No markdown. Format:
{{
  "picks": [
    {{
      "ticker": "SYMBOL",
      "bias": "bullish" | "bearish",
      "conviction": 1-10,
      "one_liner": "why this fits your style in one sentence"
    }}
  ],
  "passes_on": "brief note on why you're skipping other candidates or picking fewer than {max_picks}",
  "portfolio_notes": "any comments on existing positions (hold, trim, close)"
}}"""


# =============================================================================
# PHASE 3: Deep Dive (same as before — thesis per ticker per agent)
# =============================================================================

THESIS_SYSTEM = """You are a trading analyst. Your personality, trading style, and preferences are defined below. Stay in character — your biases are part of the analysis.

## Your Profile
{persona}

## Instructions
You will receive a deep indicator payload for a specific ticker. Produce a detailed trading thesis.

Respond in valid JSON only. No markdown. Format:
{{
  "ticker": "SYMBOL",
  "date": "YYYY-MM-DD",
  "bias": "bullish" | "bearish" | "neutral",
  "confidence": 1-10,
  "thesis": "2-4 sentence thesis explaining your read",
  "setup": {{
    "entry_trigger": "what would make you enter",
    "stop_loss": "where you'd place the stop",
    "target": "profit target",
    "timeframe": "expected hold period"
  }},
  "key_indicators": ["which indicators from the payload drove your decision"],
  "concerns": ["what could invalidate this thesis"],
  "would_trade": true | false,
  "why_not": "if would_trade is false, explain why you're sitting out"
}}"""


# =============================================================================
# PHASE 4: Red Team, Rebuttal, and Capo
# =============================================================================

RED_TEAM_SYSTEM = """You are a trading analyst performing adversarial review. Your personality and trading style are defined below. Stay in character.

## Your Profile
{persona}

## Instructions
You are reviewing another advisor's trading thesis on {ticker}. Your job is to ruthlessly attack it:
- Identify logical fallacies and contradictions
- Point out data they ignored or misinterpreted
- Challenge their assumptions about the regime
- Find risks they underweighted or missed entirely

Be specific. Reference actual indicator values when attacking.

Respond in valid JSON only. No markdown. Format:
{{
  "reviewer": "your name",
  "target": "name of advisor being reviewed",
  "verdict": "agree" | "disagree" | "partially_agree",
  "flaws": ["specific flaw 1", "specific flaw 2"],
  "missed_signals": ["indicator or signal they should have considered"],
  "risk_blind_spots": ["risks they underweighted"],
  "counter_thesis": "1-2 sentence alternative read if you disagree"
}}"""


REBUTTAL_SYSTEM = """You are a trading analyst defending your thesis against criticism. Your personality and trading style are defined below. Stay in character.

## Your Profile
{persona}

## Instructions
You produced a trading thesis on {ticker} that has been challenged by other advisors. You MUST defend your thesis. You cannot abandon or withdraw it — the Capo will decide whose thesis has the most merit.

Respond in valid JSON only. No markdown. Format:
{{
  "defender": "your name",
  "ticker": "{ticker}",
  "acknowledged_risks": ["valid points from critics that you accept"],
  "rebuttals": ["point-by-point defense"],
  "revised_confidence": 1-10,
  "strongest_argument": "your single best reason this thesis is correct"
}}"""


CAPO_SYSTEM = """You are the Capo of the Consiglio — the final decision maker. Your trading personality is informed by the persona below, but you have authority to override your tendencies when the evidence demands it.

## Your Persona
{persona}

## Your Role
You receive an executive summary of today's Consiglio deliberation. For each ticker debated, you see:
- The post-rebuttal thesis from each advisor who pitched it
- The strongest counter-arguments raised against it

Use this to make concrete portfolio allocation decisions.

## Investment Mandate (from the Principal — these override all other rules)
- You MUST deploy at least 80% of the portfolio into positions. Cash sitting idle is failure.
- Invested capital must stay between 75% and 90% of total portfolio value at all times.
- If invested % drops below 75%, you MUST buy on the next decision.
- If invested % rises above 90%, you MUST trim.
- You decide WHICH thesis to back and HOW to express it — but you must be in the market.

## Rules
- You manage a portfolio starting at $1000 with fractional shares
- You can buy or sell any ticker the advisors analyzed today
- Position sizes must be in dollar amounts
- Max 8 concurrent positions
- Evaluate which advisor's thesis survived the debate best
- If you have open positions, actively manage them (hold, add, trim, or close)

Respond in valid JSON only. No markdown. Format:
{{
  "date": "YYYY-MM-DD",
  "invested_pct": N,
  "actions": [
    {{"ticker": "SYMBOL", "action": "buy" | "sell" | "sell_all" | "hold", "amount_usd": N, "reason": "brief"}}
  ],
  "best_thesis": "name of the advisor whose thesis you found most compelling",
  "rationale": "2-3 sentence overall assessment of the day",
  "risk_notes": "any portfolio-level risk concerns"
}}"""


# =============================================================================
# HELPERS
# =============================================================================

def truncate_to_date(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    return df.loc[:as_of_date]


def get_trading_days(ticker_df: pd.DataFrame, start_date: str, end_date: str) -> list:
    mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
    return [d.strftime("%Y-%m-%d") for d in ticker_df.index[mask]]


def get_next_day_open(ticker: str, as_of_date: str) -> float | None:
    df = load_ticker(ticker)
    future = df.loc[as_of_date:]
    if len(future) < 2:
        return None
    return float(future.iloc[1]["Open"])


def get_current_prices(tickers: list, as_of_date: str) -> dict:
    prices = {}
    for t in tickers:
        try:
            df = load_ticker(t)
            truncated = truncate_to_date(df, as_of_date)
            if len(truncated) > 0:
                prices[t] = float(truncated.iloc[-1]["Close"])
        except Exception:
            pass
    return prices


def _qwen_call(system: str, user_msg: str, label: str = "") -> dict | None:
    """Generic Qwen call that returns parsed JSON or None."""
    import requests
    try:
        resp = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.4,
            "max_tokens": 2048,
        }, timeout=300)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"        {label} FAILED: {e}")
        return None


def _load_persona(name: str) -> str:
    path = os.path.join(CONSIGLIO_DIR, name, "brain.md")
    with open(path) as f:
        return f.read()


def _parse_wait_time(error_text: str, default: int = 60) -> int:
    import re
    patterns = [
        r'(\d+)\s*(?:seconds?|s)\b',
        r'retry.after.*?(\d+)',
        r'wait.*?(\d+)',
    ]
    for p in patterns:
        m = re.search(p, error_text, re.IGNORECASE)
        if m:
            return int(m.group(1)) + 5
    return default


# =============================================================================
# PHASE 2: PITCH
# =============================================================================

def _condensed_persona(name: str) -> str:
    """Extract Identity + Trading Style + Indicator Preferences sections only."""
    text = _load_persona(name)
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


def run_pitch(persona_name: str, screener_output: list, portfolio_state: str,
              as_of_date: str) -> dict | None:
    """Agent picks 0-3 tickers from screener output."""
    persona_text = _condensed_persona(persona_name)
    system = PITCH_SYSTEM.format(
        persona=persona_text, max_picks=MAX_PICKS_PER_AGENT
    )

    user_msg = f"""Date: {as_of_date}

Screener ({len(screener_output)} candidates):
{json.dumps(screener_output)}

Portfolio: {portfolio_state}

Pick 0-{MAX_PICKS_PER_AGENT} tickers."""

    return _qwen_call(system, user_msg, f"{persona_name} pitch")


# =============================================================================
# PHASE 3: DEEP DIVE
# =============================================================================

def run_deep_thesis(persona_name: str, payload: dict) -> dict | None:
    """Run a single persona's deep thesis on a specific ticker payload."""
    persona_text = _load_persona(persona_name)
    system = THESIS_SYSTEM.format(persona=persona_text)

    user_msg = f"""End-of-day payload for {payload['ticker']} on {payload['date']}:

Market Context:
{json.dumps(payload['context'])}

Available Indicators:
{json.dumps(payload['indicators'])}

Analyze this data in character and produce your trading thesis."""

    return _qwen_call(system, user_msg, f"{persona_name}→{payload['ticker']}")


# =============================================================================
# PHASE 4: DEBATE (isolated per ticker)
# =============================================================================

def run_red_team(reviewer_name: str, target_name: str, target_thesis: dict,
                 ticker: str, payload: dict) -> dict | None:
    persona_text = _load_persona(reviewer_name)
    system = RED_TEAM_SYSTEM.format(persona=persona_text, ticker=ticker)

    user_msg = f"""You are {reviewer_name}. Review {target_name}'s thesis on {ticker}:

{json.dumps(target_thesis)}

Key indicator values:
{json.dumps({k: v for k, v in list(payload['indicators'].items())[:20]})}

Attack this thesis. Find the flaws."""

    return _qwen_call(system, user_msg, f"{reviewer_name}→{target_name}({ticker})")


def run_rebuttal(defender_name: str, thesis: dict, critiques: list,
                 ticker: str) -> dict | None:
    persona_text = _load_persona(defender_name)
    system = REBUTTAL_SYSTEM.format(persona=persona_text, ticker=ticker)

    critiques_text = ""
    for c in critiques:
        if c:
            reviewer = c.get("reviewer", "unknown")
            critiques_text += f"\n--- From {reviewer} ---\n{json.dumps(c)}\n"

    user_msg = f"""You are {defender_name}. Your thesis on {ticker}:
{json.dumps(thesis)}

Criticisms:
{critiques_text}

Defend your thesis."""

    return _qwen_call(system, user_msg, f"{defender_name} rebuttal({ticker})")


def run_ticker_debate(ticker: str, pitchers: list[str], payload: dict) -> dict:
    """Run the full debate loop for a single ticker. Returns summary dict."""
    # Step 1: Each pitcher produces a deep thesis
    theses = {}
    for name in pitchers:
        t0 = time.time()
        thesis = run_deep_thesis(name, payload)
        elapsed = time.time() - t0
        theses[name] = thesis
        if thesis:
            bias = thesis.get("bias", "?")
            conf = thesis.get("confidence", "?")
            trade = thesis.get("would_trade", "?")
            print(f"          {name:10s}: {bias:8s} conf={conf} trade={trade} ({elapsed:.0f}s)")
        else:
            print(f"          {name:10s}: FAILED ({elapsed:.0f}s)")

    # Step 2: Red team — each pitcher attacks each other pitcher's thesis
    red_teams = {}
    for reviewer in pitchers:
        for target in pitchers:
            if reviewer == target or theses.get(target) is None:
                continue
            t0 = time.time()
            critique = run_red_team(reviewer, target, theses[target], ticker, payload)
            elapsed = time.time() - t0
            key = f"{reviewer}→{target}"
            red_teams[key] = critique
            if critique:
                verdict = critique.get("verdict", "?")
                print(f"          RT {key:25s}: {verdict} ({elapsed:.0f}s)")
            else:
                print(f"          RT {key:25s}: FAILED ({elapsed:.0f}s)")

    # Step 3: Rebuttals
    rebuttals = {}
    for defender in pitchers:
        if theses.get(defender) is None:
            continue
        critiques_for = [v for k, v in red_teams.items()
                         if k.endswith(f"→{defender}") and v is not None]
        if not critiques_for:
            rebuttals[defender] = None
            continue
        t0 = time.time()
        rebuttal = run_rebuttal(defender, theses[defender], critiques_for, ticker)
        elapsed = time.time() - t0
        rebuttals[defender] = rebuttal
        if rebuttal:
            new_conf = rebuttal.get("revised_confidence", "?")
            strongest = str(rebuttal.get("strongest_argument", ""))[:60]
            print(f"          RB {defender:10s}: conf={new_conf} — {strongest} ({elapsed:.0f}s)")
        else:
            print(f"          RB {defender:10s}: FAILED ({elapsed:.0f}s)")

    return {
        "ticker": ticker,
        "theses": theses,
        "red_teams": red_teams,
        "rebuttals": rebuttals,
    }


def build_executive_summary(debates: list[dict]) -> str:
    """Extract post-rebuttal theses and strongest counter-arguments.
    This is what goes to the Capo — NOT the full debate transcript."""
    sections = []
    for debate in debates:
        ticker = debate["ticker"]
        lines = [f"### {ticker}"]

        for name, rebuttal in debate["rebuttals"].items():
            thesis = debate["theses"].get(name)
            if thesis is None:
                continue

            bias = thesis.get("bias", "?")
            conf = thesis.get("confidence", "?")
            revised_conf = rebuttal.get("revised_confidence", conf) if rebuttal else conf
            strongest = rebuttal.get("strongest_argument", thesis.get("thesis", "")) if rebuttal else thesis.get("thesis", "")
            setup = thesis.get("setup", {})

            lines.append(f"\n**{name}** — {bias} (confidence {conf}→{revised_conf})")
            lines.append(f"  Thesis: {thesis.get('thesis', '?')}")
            if thesis.get("would_trade"):
                lines.append(f"  Entry: {setup.get('entry_trigger', '?')}")
                lines.append(f"  Stop: {setup.get('stop_loss', '?')}")
                lines.append(f"  Target: {setup.get('target', '?')}")
            lines.append(f"  Strongest argument: {strongest}")

            # Strongest counter-arguments against this thesis
            counters = []
            for key, critique in debate["red_teams"].items():
                if critique and key.endswith(f"→{name}"):
                    flaws = critique.get("flaws", [])
                    if flaws:
                        reviewer = critique.get("reviewer", key.split("→")[0])
                        counters.append(f"{reviewer}: {flaws[0]}")
            if counters:
                lines.append(f"  Top counter-arguments: {'; '.join(counters[:2])}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# =============================================================================
# CAPO DECISION (Claude CLI)
# =============================================================================

def run_capo(capo_persona: str, executive_summary: str,
             portfolio_state: str, as_of_date: str) -> dict | None:
    persona_text = _load_persona(capo_persona)
    system = CAPO_SYSTEM.format(persona=persona_text)

    user_msg = f"""Date: {as_of_date}

## Executive Summary — Today's Consiglio Deliberation
{executive_summary}

## Current Portfolio
{portfolio_state}

Make your trading decisions for today."""

    full_prompt = f"{system}\n\n{user_msg}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["claude", "-p", "--output-format", "json", "--model", "sonnet-4-6", "--effort", "medium"],
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

    print(f"      Max retries exceeded for Claude")
    return None


# =============================================================================
# TRADE EXECUTION
# =============================================================================

def execute_actions(portfolio: Portfolio, actions: list, as_of_date: str):
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


# =============================================================================
# MAIN BACKTEST LOOP
# =============================================================================

def run_backtest(start_date: str, end_date: str,
                 capo: str = "Lorenzo", starting_cash: float = 1000.0):
    run_id = f"consiglio_{start_date}_{end_date}_{capo}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MEDICI BACKTEST v3 — Full Consiglio")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Capo:   {capo}")
    print(f"  Agents: {', '.join(PERSONAS)}")
    print(f"  Cash:   ${starting_cash:.2f}")
    print(f"{'='*70}")

    # Use SPY as the calendar
    spy_df = load_ticker("SPY")
    trading_days = get_trading_days(spy_df, start_date, end_date)
    print(f"\n  {len(trading_days)} trading days\n")

    # Pre-load cross-asset data
    full_vix = load_vix()
    full_spy = load_ticker("SPY")
    full_hyg = load_optional("HYG")
    full_tlt = load_optional("TLT")

    portfolio = Portfolio(starting_cash=starting_cash, log_dir=run_dir)

    for day_num, date in enumerate(trading_days, 1):
        print(f"\n  DAY {day_num}/{len(trading_days)} — {date}")
        print(f"  {'-'*60}")

        # Current portfolio state for prompts
        all_held = list(portfolio.positions.keys())
        prices = get_current_prices(all_held + ["SPY"], date)
        portfolio_state = portfolio.get_state_for_prompt(prices)

        # =====================================================================
        # PHASE 1: SCREENER (pure Python, no LLM)
        # =====================================================================
        print(f"    [P1] Screener...")
        t0 = time.time()
        candidates = run_screener(date, spy_df=full_spy)
        elapsed = time.time() - t0
        print(f"         {len(candidates)} candidates in {elapsed:.1f}s")

        if not candidates:
            print(f"         No candidates today, skipping to portfolio management")
            # Still need Capo to manage existing positions
            if portfolio.positions:
                summary = "No screener candidates today. Review existing positions only."
                decision = run_capo(capo, summary, portfolio_state, date)
                if decision:
                    execute_actions(portfolio, decision.get("actions", []), date)
            snap = portfolio.snapshot(date, get_current_prices(all_held, date))
            print(f"    [--] Portfolio: ${snap['total_value']:.2f} ({snap['total_return_pct']:+.2f}%)")
            portfolio.save()
            _save_day(run_dir, day_num, date, {"screener": [], "pitches": {},
                      "debates": [], "decision": None, "portfolio_snapshot": snap})
            continue

        # Cap screener output sent to agents (top 15 by score, compact format)
        screener_for_agents = [
            {k: c[k] for k in ("ticker", "adx", "rsi", "price_vs_200sma_pct", "rvol", "zscore_50")}
            for c in candidates[:15]
        ]

        # =====================================================================
        # PHASE 2: PITCH (each agent picks 0-3 tickers)
        # =====================================================================
        print(f"    [P2] Pitches:")
        pitches = {}  # {persona: pitch_result}
        ticker_advocates = {}  # {ticker: [persona names who pitched it]}

        for persona in PERSONAS:
            t0 = time.time()
            pitch = run_pitch(persona, screener_for_agents, portfolio_state, date)
            elapsed = time.time() - t0
            pitches[persona] = pitch

            if pitch and pitch.get("picks"):
                picks = pitch["picks"][:MAX_PICKS_PER_AGENT]  # enforce hard cap
                tickers_picked = [p["ticker"] for p in picks]
                for p in picks:
                    t_name = p["ticker"]
                    ticker_advocates.setdefault(t_name, []).append(persona)
                print(f"         {persona:10s}: {', '.join(tickers_picked)} ({elapsed:.0f}s)")
            else:
                print(f"         {persona:10s}: no picks ({elapsed:.0f}s)")

        # Deduplicate
        unique_tickers = sorted(ticker_advocates.keys())
        print(f"         Unique tickers pitched: {len(unique_tickers)} — {unique_tickers}")

        if not unique_tickers:
            print(f"         No one pitched anything today")
            if portfolio.positions:
                summary = "No advisor pitched any tickers today. Review existing positions only."
                decision = run_capo(capo, summary, portfolio_state, date)
                if decision:
                    execute_actions(portfolio, decision.get("actions", []), date)
            snap = portfolio.snapshot(date, get_current_prices(all_held, date))
            print(f"    [--] Portfolio: ${snap['total_value']:.2f} ({snap['total_return_pct']:+.2f}%)")
            portfolio.save()
            _save_day(run_dir, day_num, date, {"screener": candidates[:10], "pitches": pitches,
                      "debates": [], "decision": None, "portfolio_snapshot": snap})
            continue

        # =====================================================================
        # PHASE 3: DEEP DIVES (heavy indicators + scoring, once per ticker)
        # =====================================================================
        print(f"    [P3] Deep dives:")
        payloads = {}  # {ticker: payload}

        for ticker in unique_tickers:
            print(f"         --- {ticker} ---")
            try:
                t_df = load_ticker(ticker)
                t_df = truncate_to_date(t_df, date)
                if len(t_df) < 200:
                    print(f"         {ticker}: insufficient data ({len(t_df)} rows), skipping")
                    continue

                spy_trunc = truncate_to_date(full_spy, date)
                vix_trunc = truncate_to_date(full_vix, date) if full_vix is not None else None
                hyg_trunc = truncate_to_date(full_hyg, date) if full_hyg is not None else None
                tlt_trunc = truncate_to_date(full_tlt, date) if full_tlt is not None else None

                # Compute indicators
                t0 = time.time()
                all_ind = compute_all(
                    t_df, spy_df=spy_trunc, hyg_df=hyg_trunc, tlt_df=tlt_trunc,
                )
                computed = sum(1 for v in all_ind.values() if v is not None)

                # Build context
                ctx = build_context(ticker, t_df, vix_trunc)

                # Score indicators via Qwen
                scoring = score_indicators(ctx)

                # Pack payload
                payload = pack_payload(ticker, date, ctx, all_ind, scoring)
                payloads[ticker] = payload
                elapsed = time.time() - t0
                kept = payload["scoring_summary"]["kept"]
                print(f"         {ticker}: {computed} indicators → {kept} kept ({elapsed:.0f}s)")

            except Exception as e:
                print(f"         {ticker}: ERROR — {e}")

        if not payloads:
            print(f"         No payloads generated")
            snap = portfolio.snapshot(date, get_current_prices(all_held, date))
            portfolio.save()
            _save_day(run_dir, day_num, date, {"screener": candidates[:10], "pitches": pitches,
                      "debates": [], "decision": None, "portfolio_snapshot": snap})
            continue

        # =====================================================================
        # PHASE 4a: ISOLATED DEBATES (per ticker)
        # =====================================================================
        print(f"    [P4a] Debates:")
        debates = []

        for ticker, payload in payloads.items():
            advocates = ticker_advocates.get(ticker, [])
            if not advocates:
                continue
            print(f"         --- {ticker} (advocates: {', '.join(advocates)}) ---")
            debate = run_ticker_debate(ticker, advocates, payload)
            debates.append(debate)

        # =====================================================================
        # PHASE 4b: EXECUTIVE SUMMARY → CAPO
        # =====================================================================
        print(f"    [P4b] Capo ({capo}) deciding...")
        exec_summary = build_executive_summary(debates)

        # Add existing position context
        if portfolio.positions:
            exec_summary += f"\n\n### Existing Positions\n"
            for t, pos in portfolio.positions.items():
                p = prices.get(t, pos["avg_cost"])
                pnl_pct = (p / pos["avg_cost"] - 1) * 100 if pos["avg_cost"] > 0 else 0
                exec_summary += f"  {t}: {pos['shares']:.4f} @ ${pos['avg_cost']:.2f} → ${p:.2f} ({pnl_pct:+.1f}%)\n"

        t0 = time.time()
        decision = run_capo(capo, exec_summary, portfolio_state, date)
        elapsed = time.time() - t0
        print(f"         Decision in {elapsed:.0f}s")

        if decision:
            actions = decision.get("actions", [])
            rationale = decision.get("rationale", "")
            print(f"         Rationale: {rationale}")
            if actions:
                print(f"         Actions ({len(actions)}):")
                for a in actions:
                    print(f"           {a.get('action','?').upper()} {a.get('ticker','?')} ${a.get('amount_usd',0):.2f} — {a.get('reason','')}")
                execute_actions(portfolio, actions, date)
            else:
                print(f"         No trades today")
        else:
            print(f"         Capo returned no decision")

        # Daily snapshot
        all_held_now = list(portfolio.positions.keys())
        prices_now = get_current_prices(all_held_now, date)
        snap = portfolio.snapshot(date, prices_now)
        print(f"    [>>] Portfolio: ${snap['total_value']:.2f} ({snap['total_return_pct']:+.2f}%)")

        portfolio.save()
        _save_day(run_dir, day_num, date, {
            "screener_count": len(candidates),
            "screener_top10": candidates[:10],
            "pitches": pitches,
            "unique_tickers": unique_tickers,
            "payloads_generated": list(payloads.keys()),
            "debates": [{
                "ticker": d["ticker"],
                "theses": d["theses"],
                "red_teams": d["red_teams"],
                "rebuttals": d["rebuttals"],
            } for d in debates],
            "executive_summary": exec_summary,
            "decision": decision,
            "portfolio_snapshot": snap,
        })

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
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
        print(f"  Peak Value:   ${peak:.2f}")

    print(f"\n  Results saved to: {run_dir}")
    portfolio.save()


def _save_day(run_dir, day_num, date, data):
    day_path = os.path.join(run_dir, f"day_{day_num:03d}_{date}.json")
    with open(day_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-03-10"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-03-14"
    capo = sys.argv[3] if len(sys.argv) > 3 else "Lorenzo"

    run_backtest(start, end, capo=capo)
