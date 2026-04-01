"""
Office di Capo — Ticker Deep Dive

On-demand analysis when the Capo (or user) requests a specific ticker review.
Triggers all departments for that single ticker, then synthesizes.
"""

import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from schemas import TickerDeepDive, TradeDirective, Conviction, Action
from Headquarters.Office_di_Capo.capo_daily import (
    load_persona, load_bookshelf, llm_call, run_macro_analysis,
)


def run_ticker(ticker: str, as_of_date: str) -> TickerDeepDive | None:
    """Deep dive on a single ticker — all departments report, Capo decides."""
    print(f"\n{'='*70}")
    print(f"OFFICE DI CAPO — Ticker Deep Dive: {ticker}")
    print(f"  Date: {as_of_date}")
    print(f"{'='*70}")

    persona = load_persona("Giovanni")

    # Macro context
    print(f"\n  [1] Macro context...")
    macro = run_macro_analysis(as_of_date, persona)

    # Department reports for this ticker
    print(f"\n  [2] Department reports...")
    reports = {}

    try:
        from Headquarters.Dpt_of_Technical_Analysis.technical_ticker import run_ticker as tech_ticker
        reports["technical"] = tech_ticker(ticker, as_of_date)
    except ImportError:
        print(f"      Technical: not operational")

    try:
        from Headquarters.Dpt_of_Risk_Management.risk_ticker import run_ticker as risk_ticker
        reports["risk"] = risk_ticker(ticker, as_of_date)
    except ImportError:
        print(f"      Risk: not operational")

    try:
        from Headquarters.Dpt_of_Sentiment_Analysis.sentiment_ticker import run_ticker as sent_ticker
        reports["sentiment"] = sent_ticker(ticker, as_of_date)
    except ImportError:
        print(f"      Sentiment: not operational")

    # Capo verdict
    print(f"\n  [3] Capo verdict...")
    dept_data = json.dumps(reports, indent=2, default=str)
    macro_data = macro.model_dump_json(indent=2) if macro else "{}"

    system = f"""{persona['core_directive']}

You are reviewing {ticker} in detail. Your macro read and department reports are below.
Produce your trading verdict.

Respond in valid JSON only. Schema:
{{
    "ticker": "{ticker}",
    "action": "buy|sell|sell_all|hold|short|cover",
    "amount_usd": N,
    "conviction": "high|medium|low",
    "reason": "2-3 sentences",
    "stop_loss": N or null,
    "take_profit": N or null
}}"""

    user_msg = f"""Macro: {macro_data}
Department Reports: {dept_data}
Your verdict on {ticker}?"""

    result = llm_call(system, user_msg, persona, label=f"verdict({ticker})")
    if result:
        try:
            verdict = TradeDirective(**result)
            print(f"      {verdict.action.upper()} {ticker} ${verdict.amount_usd:.2f} [{verdict.conviction}]")
            print(f"      {verdict.reason}")
            return verdict
        except Exception as e:
            print(f"      Schema validation failed: {e}")

    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python capo_ticker.py <ticker> [date]")
        sys.exit(1)
    ticker = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else "2025-03-10"
    run_ticker(ticker, date)
