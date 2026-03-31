"""
Medici — Backtest Engine

Runs the full Consiglio pipeline day-by-day on historical data.
The Capo (Claude) makes trading decisions based on persona theses.
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
from pipeline import (
    load_ticker, load_optional, load_universe, load_sectors,
    score_indicators, pack_payload, DATA_DIR
)

CONSIGLIO_DIR = os.path.join(os.path.dirname(__file__), "Consiglio")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "backtests")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

PERSONAS = ["Cosimo", "Lorenzo", "Giuliano"]

AGENT_SYSTEM = """You are a trading analyst. Your personality, trading style, and preferences are defined below. Stay in character — your biases are part of the analysis.

## Your Profile
{persona}

## Instructions
You will receive an end-of-day market data payload for a ticker. Analyze it and produce a trading thesis.

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


RED_TEAM_SYSTEM = """You are a trading analyst performing adversarial review. Your personality and trading style are defined below. Stay in character.

## Your Profile
{persona}

## Instructions
You are reviewing another advisor's trading thesis. Your job is to ruthlessly attack it:
- Identify logical fallacies and contradictions
- Point out data they ignored or misinterpreted
- Challenge their assumptions about the regime
- Find risks they underweighted or missed entirely
- Question whether their indicators actually support their conclusion

Be specific. Reference actual indicator values from the payload when attacking.

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
You produced a trading thesis that has been challenged by other advisors. You MUST defend your thesis. You cannot abandon or withdraw it — the Capo will decide whose thesis has the most merit. Acknowledge valid criticisms but explain why your core thesis still holds despite them.

Respond in valid JSON only. No markdown. Format:
{{
  "defender": "your name",
  "acknowledged_risks": ["valid points from critics that you accept as real risks"],
  "rebuttals": ["point-by-point defense of your thesis"],
  "revised_confidence": 1-10,
  "strongest_argument": "your single best reason this thesis is correct"
}}"""


CAPO_SYSTEM = """You are the Capo of the Consiglio — the final decision maker. Your trading personality is defined by the persona below, but you are not bound rigidly to it. You have the authority to override your natural tendencies when the evidence demands it.

## Your Persona
{persona}

## Your Role
You receive the full Consiglio deliberation: initial theses from three advisors, their red-team critiques of each other, and their rebuttals to criticism. Use this complete debate to make concrete trading decisions.

## Investment Mandate (from the Principal — these override all other rules)
- You MUST deploy at least 80% of the portfolio into positions. Cash sitting idle is failure.
- Invested capital must stay between 75% and 90% of total portfolio value at all times.
- If invested % drops below 75%, you MUST buy on the next decision.
- If invested % rises above 90%, you MUST trim.
- You decide WHICH thesis to back and HOW to express it — but you must be in the market.

## Rules
- You manage a portfolio starting at $1000 with fractional shares
- You can buy or sell any ticker the advisors analyzed
- Position sizes must be in dollar amounts
- You MUST respect the risk limits of your persona for individual positions
- Evaluate which advisor's thesis survived the red-team debate best and weight your decisions accordingly
- If you have open positions, actively manage them (hold, add, trim, or close)

Respond in valid JSON only. No markdown. Format:
{{
  "date": "YYYY-MM-DD",
  "invested_pct": N,
  "actions": [
    {{"ticker": "SYMBOL", "action": "buy" | "sell" | "sell_all" | "hold", "amount_usd": N, "reason": "brief"}}
  ],
  "best_thesis": "name of the advisor whose thesis you found most compelling after the debate",
  "rationale": "2-3 sentence overall assessment of the day",
  "risk_notes": "any portfolio-level risk concerns"
}}"""


def truncate_to_date(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Return df with only rows up to and including as_of_date."""
    return df.loc[:as_of_date]


def get_trading_days(ticker_df: pd.DataFrame, start_date: str, end_date: str) -> list:
    """Get list of trading days between start and end from actual data."""
    mask = (ticker_df.index >= start_date) & (ticker_df.index <= end_date)
    return [d.strftime("%Y-%m-%d") for d in ticker_df.index[mask]]


def get_next_day_open(ticker: str, as_of_date: str) -> float | None:
    """Get the open price on the next trading day after as_of_date."""
    df = load_ticker(ticker)
    future = df.loc[as_of_date:]
    if len(future) < 2:
        return None
    return float(future.iloc[1]["Open"])


def get_current_prices(tickers: list, as_of_date: str) -> dict:
    """Get close prices for multiple tickers on as_of_date."""
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


def run_persona(persona_name: str, payload: dict) -> dict | None:
    """Run a single persona through Qwen. Returns thesis dict or None."""
    brain_path = os.path.join(CONSIGLIO_DIR, persona_name, "brain.md")
    with open(brain_path) as f:
        persona_text = f.read()

    system = AGENT_SYSTEM.format(persona=persona_text)

    compact_ind = {}
    for k, v in payload["indicators"].items():
        compact_ind[k] = v

    user_msg = f"""End-of-day payload for {payload['ticker']} on {payload['date']}:

Market Context:
{json.dumps(payload['context'])}

Available Indicators:
{json.dumps(compact_ind)}

Analyze this data in character and produce your trading thesis."""

    try:
        import requests
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
        print(f"      {persona_name} FAILED: {e}")
        return None


def _qwen_call(system: str, user_msg: str, label: str = "") -> dict | None:
    """Generic Qwen call that returns parsed JSON or None."""
    import requests as req
    try:
        resp = req.post(QWEN_API, json={
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


def run_red_team(reviewer_name: str, target_name: str, target_thesis: dict,
                 payload: dict) -> dict | None:
    """One persona red-teams another's thesis."""
    brain_path = os.path.join(CONSIGLIO_DIR, reviewer_name, "brain.md")
    with open(brain_path) as f:
        persona_text = f.read()

    system = RED_TEAM_SYSTEM.format(persona=persona_text)
    user_msg = f"""You are {reviewer_name}. Review {target_name}'s thesis:

{json.dumps(target_thesis)}

Market context for reference:
{json.dumps(payload['context'])}

Key indicator values:
{json.dumps({k: v for k, v in list(payload['indicators'].items())[:30]})}

Attack this thesis. Find the flaws."""

    return _qwen_call(system, user_msg, f"{reviewer_name}→{target_name}")


def run_rebuttal(defender_name: str, thesis: dict, critiques: list) -> dict | None:
    """Persona defends their thesis against red team attacks."""
    brain_path = os.path.join(CONSIGLIO_DIR, defender_name, "brain.md")
    with open(brain_path) as f:
        persona_text = f.read()

    system = REBUTTAL_SYSTEM.format(persona=persona_text)

    critiques_text = ""
    for c in critiques:
        if c:
            reviewer = c.get("reviewer", "unknown")
            critiques_text += f"\n--- From {reviewer} ---\n{json.dumps(c)}\n"

    user_msg = f"""You are {defender_name}. Your thesis:
{json.dumps(thesis)}

Criticisms from other advisors:
{critiques_text}

Defend your thesis or concede where they're right."""

    return _qwen_call(system, user_msg, f"{defender_name} rebuttal")


def run_capo(capo_persona: str, theses: dict, red_teams: dict, rebuttals: dict,
             portfolio_state: str, as_of_date: str, ticker: str) -> dict | None:
    """Run the Capo decision through Claude CLI. Handles rate limits."""
    brain_path = os.path.join(CONSIGLIO_DIR, capo_persona, "brain.md")
    with open(brain_path) as f:
        persona_text = f.read()

    system = CAPO_SYSTEM.format(persona=persona_text)

    # Build the full debate transcript
    debate = ""
    for name in theses:
        thesis = theses.get(name)
        debate += f"\n### {name} — THESIS\n"
        if thesis:
            debate += json.dumps(thesis) + "\n"
        else:
            debate += "Failed to produce thesis.\n"

    debate += "\n## RED TEAM CRITIQUES\n"
    for key, critique in red_teams.items():
        debate += f"\n### {key}\n"
        if critique:
            debate += json.dumps(critique) + "\n"
        else:
            debate += "No critique produced.\n"

    debate += "\n## REBUTTALS\n"
    for name, rebuttal in rebuttals.items():
        debate += f"\n### {name} responds\n"
        if rebuttal:
            debate += json.dumps(rebuttal) + "\n"
        else:
            debate += "No rebuttal produced.\n"

    user_msg = f"""Date: {as_of_date}
Ticker under analysis: {ticker}

## Consiglio Deliberation
{debate}

## Current Portfolio
{portfolio_state}

Make your trading decisions for today."""

    full_prompt = f"{system}\n\n{user_msg}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["claude", "-p", "--output-format", "json"],
                input=full_prompt,
                capture_output=True, text=True, timeout=120,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Rate limit detection
                if "rate" in stderr.lower() or "429" in stderr or "limit" in stderr.lower() or "overloaded" in stderr.lower():
                    wait = _parse_wait_time(stderr, default=60)
                    print(f"      Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                print(f"      Claude error: {stderr[:200]}")
                return None

            output = json.loads(result.stdout)
            raw = output.get("result", "")

            # Parse JSON from Claude's response
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            cleaned = cleaned.strip()

            # Find JSON in response (Claude might add text around it)
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
            if "rate" in stderr.lower() or "429" in stderr or "limit" in stderr.lower():
                wait = _parse_wait_time(stderr, default=60)
                print(f"      Rate limited. Waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"      Claude error: {e}")
            return None

    print(f"      Max retries exceeded for Claude")
    return None


def _parse_wait_time(error_text: str, default: int = 60) -> int:
    """Try to extract wait time from rate limit error message."""
    import re
    # Look for patterns like "try again in 30s" or "retry after 60 seconds"
    patterns = [
        r'(\d+)\s*(?:seconds?|s)\b',
        r'retry.after.*?(\d+)',
        r'wait.*?(\d+)',
    ]
    for p in patterns:
        m = re.search(p, error_text, re.IGNORECASE)
        if m:
            return int(m.group(1)) + 5  # add buffer
    return default


def execute_actions(portfolio: Portfolio, actions: list, as_of_date: str):
    """Execute the Capo's trading decisions at next day's open."""
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

        # Get the next trading day's date for the trade log
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
                pos_shares = 0  # already sold
                print(f"      EXEC: SELL ALL {ticker} @ ${next_open:.2f}")


def run_backtest(ticker: str, start_date: str, end_date: str,
                 capo: str = "Lorenzo", starting_cash: float = 1000.0):
    """Run the full backtest loop."""
    run_id = f"{ticker}_{start_date}_{end_date}_{capo}"
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MEDICI BACKTEST")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {start_date} → {end_date}")
    print(f"  Capo:   {capo}")
    print(f"  Cash:   ${starting_cash:.2f}")
    print(f"{'='*70}")

    # Load data
    spy_df = load_ticker("SPY")
    trading_days = get_trading_days(spy_df, start_date, end_date)
    print(f"\n  {len(trading_days)} trading days\n")

    # Load cross-asset data once (full, truncation happens per day)
    full_df = load_ticker(ticker)
    full_spy = load_ticker("SPY") if ticker != "SPY" else full_df
    full_vix = load_vix()
    full_hyg = load_optional("HYG")
    full_tlt = load_optional("TLT")

    portfolio = Portfolio(starting_cash=starting_cash, log_dir=run_dir)

    for day_num, date in enumerate(trading_days, 1):
        print(f"\n  DAY {day_num}/{len(trading_days)} — {date}")
        print(f"  {'-'*50}")

        # Truncate all data to this date
        df = truncate_to_date(full_df, date)
        spy_trunc = truncate_to_date(full_spy, date)
        vix_trunc = truncate_to_date(full_vix, date) if full_vix is not None else None
        hyg_trunc = truncate_to_date(full_hyg, date) if full_hyg is not None else None
        tlt_trunc = truncate_to_date(full_tlt, date) if full_tlt is not None else None

        if len(df) < 200:
            print(f"    Not enough data ({len(df)} rows), skipping")
            continue

        # 1. Compute indicators
        print(f"    [1] Computing indicators...")
        t0 = time.time()
        all_indicators = compute_all(
            df, spy_df=spy_trunc, hyg_df=hyg_trunc, tlt_df=tlt_trunc,
        )
        computed = sum(1 for v in all_indicators.values() if v is not None)
        print(f"        {computed}/{len(all_indicators)} in {time.time()-t0:.1f}s")

        # 2. Build context
        ctx = build_context(ticker, df, vix_trunc)
        print(f"    [2] Regime: {ctx['regime']}")

        # 3. Score indicators
        print(f"    [3] Scoring via Qwen...")
        t0 = time.time()
        scoring = score_indicators(ctx)
        print(f"        Scored in {time.time()-t0:.0f}s")

        # 4. Pack payload
        payload = pack_payload(ticker, date, ctx, all_indicators, scoring)
        kept = payload["scoring_summary"]["kept"]
        print(f"    [4] Payload: {kept} indicators")

        # 5. Run Consiglio — Theses
        print(f"    [5] Consiglio — Theses:")
        theses = {}
        for persona in PERSONAS:
            t0 = time.time()
            thesis = run_persona(persona, payload)
            elapsed = time.time() - t0
            if thesis:
                bias = thesis.get("bias", "?")
                trade = thesis.get("would_trade", "?")
                conf = thesis.get("confidence", "?")
                print(f"        {persona:10s}: {bias:8s} conf={conf} trade={trade} ({elapsed:.0f}s)")
            else:
                print(f"        {persona:10s}: FAILED ({elapsed:.0f}s)")
            theses[persona] = thesis

        # 6. Red Team — Each persona critiques the other two
        print(f"    [6] Red Team:")
        red_teams = {}
        for reviewer in PERSONAS:
            for target in PERSONAS:
                if reviewer == target:
                    continue
                if theses.get(target) is None:
                    continue
                t0 = time.time()
                critique = run_red_team(reviewer, target, theses[target], payload)
                elapsed = time.time() - t0
                key = f"{reviewer}→{target}"
                red_teams[key] = critique
                if critique:
                    verdict = critique.get("verdict", "?")
                    n_flaws = len(critique.get("flaws", []))
                    print(f"        {key:25s}: {verdict} ({n_flaws} flaws) ({elapsed:.0f}s)")
                else:
                    print(f"        {key:25s}: FAILED ({elapsed:.0f}s)")

        # 7. Rebuttals — Each persona defends against criticism
        print(f"    [7] Rebuttals:")
        rebuttals = {}
        for defender in PERSONAS:
            if theses.get(defender) is None:
                continue
            # Collect critiques aimed at this defender
            critiques_for = [v for k, v in red_teams.items()
                            if k.endswith(f"→{defender}") and v is not None]
            if not critiques_for:
                rebuttals[defender] = None
                continue
            t0 = time.time()
            rebuttal = run_rebuttal(defender, theses[defender], critiques_for)
            elapsed = time.time() - t0
            rebuttals[defender] = rebuttal
            if rebuttal:
                new_conf = rebuttal.get("revised_confidence", "?")
                n_risks = len(rebuttal.get("acknowledged_risks", []))
                strongest = rebuttal.get("strongest_argument", "")[:80]
                print(f"        {defender:10s}: conf={new_conf} acks={n_risks} — {strongest} ({elapsed:.0f}s)")
            else:
                print(f"        {defender:10s}: FAILED ({elapsed:.0f}s)")

        # 8. Capo decides
        prices = get_current_prices(
            [ticker] + list(portfolio.positions.keys()), date
        )
        portfolio_state = portfolio.get_state_for_prompt(prices)

        print(f"    [8] Capo ({capo}) deciding...")
        t0 = time.time()
        decision = run_capo(capo, theses, red_teams, rebuttals,
                           portfolio_state, date, ticker)
        elapsed = time.time() - t0
        print(f"        Decision in {elapsed:.0f}s")

        if decision:
            actions = decision.get("actions", [])
            rationale = decision.get("rationale", "")
            print(f"        Rationale: {rationale}")
            if actions:
                print(f"        Actions ({len(actions)}):")
                for a in actions:
                    print(f"          {a.get('action','?').upper()} {a.get('ticker','?')} ${a.get('amount_usd',0):.2f} — {a.get('reason','')}")
                execute_actions(portfolio, actions, date)
            else:
                print(f"        No trades today")
        else:
            print(f"        Capo returned no decision")

        # 9. Daily snapshot
        prices = get_current_prices(
            [ticker] + list(portfolio.positions.keys()), date
        )
        snap = portfolio.snapshot(date, prices)
        print(f"    [9] Portfolio: ${snap['total_value']:.2f} ({snap['total_return_pct']:+.2f}%)")

        # Save state after each day
        portfolio.save()

        # Save day details
        day_data = {
            "date": date,
            "context": ctx,
            "scoring_kept": kept,
            "theses": theses,
            "red_teams": red_teams,
            "rebuttals": rebuttals,
            "decision": decision,
            "portfolio_snapshot": snap,
        }
        day_path = os.path.join(run_dir, f"day_{day_num:03d}_{date}.json")
        with open(day_path, "w") as f:
            json.dump(day_data, f, indent=2, default=str)

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
        print(f"  Peak Value:   ${peak:.2f}")

    print(f"\n  Results saved to: {run_dir}")
    portfolio.save()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    start = sys.argv[2] if len(sys.argv) > 2 else "2025-03-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-03-31"
    capo = sys.argv[4] if len(sys.argv) > 4 else "Lorenzo"

    run_backtest(ticker, start, end, capo=capo)
