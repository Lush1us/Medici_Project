"""
Office di Capo — Daily Loop

The main orchestrator. Runs each morning:
1. Load Capo persona (Giovanni)
2. Run macro analysis using Capo's own tools
3. Trigger each department's daily report
4. Synthesize: persona → macro → reports → final directives
5. Output final report (to user or Claude for execution)
"""

import json
import os
import sys
import time
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from schemas import (
    MacroAnalysis, CapoReport, TradeDirective,
    TechnicalReport, RiskReport, SentimentReport,
    Regime, Action, Conviction,
)

CONSIGLIO_DIR = os.path.join(PROJECT_ROOT, "Consiglio")
HQ_DIR = os.path.join(PROJECT_ROOT, "Headquarters")
LEDGER_DIR = "/mnt/iarvis/Library/ledgers/Medici"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "eod")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"


# =============================================================================
# PERSONA LOADING
# =============================================================================

def load_persona(name: str) -> dict:
    """Load a persona's YAML config."""
    path = os.path.join(CONSIGLIO_DIR, name, f"{name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def load_bookshelf(persona: dict) -> dict:
    """Load persona-specific resources from their bookshelf."""
    shelf_path = persona.get("meta", {}).get("bookshelf")
    if not shelf_path or not os.path.exists(shelf_path):
        return {}

    resources = {}
    for section in os.listdir(shelf_path):
        section_path = os.path.join(shelf_path, section)
        if os.path.isdir(section_path):
            files = []
            for f in os.listdir(section_path):
                fp = os.path.join(section_path, f)
                if os.path.isfile(fp):
                    files.append(fp)
            resources[section] = files
    return resources


# =============================================================================
# LLM INTERFACE
# =============================================================================

def llm_call(system: str, user_msg: str, persona: dict,
             schema_hint: str = "", label: str = "") -> dict | None:
    """Call the LLM with persona hyperparameters."""
    import requests

    hyper = persona.get("hyperparameters", {})
    temperature = hyper.get("temperature", 0.7)
    top_p = hyper.get("top_p", 0.9)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 1024,
        }, timeout=120)
        resp.raise_for_status()

        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end])

        print(f"    [{label}] No JSON found in response")
        return None

    except Exception as e:
        print(f"    [{label}] FAILED: {e}")
        return None


# =============================================================================
# MACRO ANALYSIS (Capo's own work)
# =============================================================================

def run_macro_analysis(as_of_date: str, persona: dict) -> MacroAnalysis | None:
    """Capo runs his own macro analysis using office tools."""
    from Headquarters.Office_di_Capo.tools.macro_scanner import scan_macro

    macro_data = scan_macro(as_of_date)

    system = f"""{persona['core_directive']}

You are performing your daily macro regime assessment.
Analyze the data and produce your macro call.

Respond in valid JSON only. No markdown. Schema:
{{
    "date": "{as_of_date}",
    "regime": "risk-on" | "risk-off" | "transitional",
    "confidence": 1-10,
    "vix_read": "brief",
    "breadth_read": "brief",
    "credit_read": "brief",
    "reasoning": "2-3 sentences",
    "sectors_favored": ["sector1"],
    "sectors_avoid": ["sector1"]
}}"""

    user_msg = f"Date: {as_of_date}\nMacro Data:\n{json.dumps(macro_data, indent=2)}"

    result = llm_call(system, user_msg, persona, label="macro")
    if result:
        try:
            return MacroAnalysis(**result)
        except Exception as e:
            print(f"    [macro] Schema validation failed: {e}")
    return None


# =============================================================================
# DEPARTMENT DISPATCH
# =============================================================================

def collect_department_reports(as_of_date: str, tickers: list[str]) -> dict:
    """Trigger each department's daily report and collect results."""
    reports = {
        "technical": [],
        "risk": [],
        "sentiment": [],
    }

    # Technical Analysis
    try:
        from Headquarters.Dpt_of_Technical_Analysis.technical_report import run_daily as tech_daily
        for ticker in tickers:
            report = tech_daily(ticker, as_of_date)
            if report:
                reports["technical"].append(report)
    except ImportError:
        print("    [dispatch] Technical Analysis department not operational")

    # Risk Management
    try:
        from Headquarters.Dpt_of_Risk_Management.risk_report import run_daily as risk_daily
        for ticker in tickers:
            report = risk_daily(ticker, as_of_date)
            if report:
                reports["risk"].append(report)
    except ImportError:
        print("    [dispatch] Risk Management department not operational")

    # Sentiment Analysis
    try:
        from Headquarters.Dpt_of_Sentiment_Analysis.sentiment_report import run_daily as sent_daily
        for ticker in tickers:
            report = sent_daily(ticker, as_of_date)
            if report:
                reports["sentiment"].append(report)
    except ImportError:
        print("    [dispatch] Sentiment Analysis department not operational")

    return reports


# =============================================================================
# SYNTHESIS
# =============================================================================

def synthesize(as_of_date: str, persona: dict, macro: MacroAnalysis,
               reports: dict, portfolio_state: str) -> CapoReport | None:
    """Capo synthesizes everything into a final report with directives.

    Synthesis order (per Giovanni.yaml):
    1. Persona — who I am, my constraints, my psychology
    2. Macro — my own regime read
    3. Reports — what the departments are telling me
    """
    constraints = persona.get("constraints", {})

    # Format department reports for the prompt
    dept_summary = []
    for dept, dept_reports in reports.items():
        if dept_reports:
            dept_summary.append(f"\n## {dept.title()} Department")
            for r in dept_reports:
                if isinstance(r, dict):
                    dept_summary.append(json.dumps(r, indent=2, default=str))
                else:
                    dept_summary.append(r.model_dump_json(indent=2))

    system = f"""{persona['core_directive']}

## Your Constraints
- Max drawdown: {constraints.get('max_drawdown', 0.15) * 100:.0f}%
- Max leverage: {constraints.get('max_leverage', 3.0)}x
- Min liquidity: {constraints.get('min_liquidity', 0.05) * 100:.0f}% cash

## Your Macro Read
{macro.model_dump_json(indent=2)}

## Department Reports
{''.join(dept_summary) if dept_summary else 'No department reports available.'}

## Current Portfolio
{portfolio_state}

Based on all of the above, produce your final trading directives for today.

Respond in valid JSON only. No markdown. Schema:
{{
    "date": "{as_of_date}",
    "capo": "Giovanni",
    "macro": <your macro analysis>,
    "portfolio_value": N,
    "invested_pct": N,
    "directives": [
        {{"ticker": "SYM", "action": "buy|sell|sell_all|hold|short|cover",
          "amount_usd": N, "conviction": "high|medium|low",
          "reason": "brief", "stop_loss": N, "take_profit": N}}
    ],
    "rationale": "2-3 sentences",
    "risk_notes": "portfolio-level concerns"
}}"""

    user_msg = f"Date: {as_of_date}. Produce your final report."

    result = llm_call(system, user_msg, persona, label="synthesis")
    if result:
        try:
            return CapoReport(**result)
        except Exception as e:
            print(f"    [synthesis] Schema validation failed: {e}")
            # Return raw result for debugging
            print(f"    [synthesis] Raw: {json.dumps(result, indent=2, default=str)[:500]}")
    return None


# =============================================================================
# LEDGER
# =============================================================================

def write_ledger(report: CapoReport):
    """Append the Capo's final report to the ledger."""
    ledger_path = os.path.join(LEDGER_DIR, f"{report.date}.json")
    with open(ledger_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"    Ledger written: {ledger_path}")


# =============================================================================
# MAIN
# =============================================================================

def run_daily(as_of_date: str, tickers: list[str] = None,
              mode: str = "manual") -> CapoReport | None:
    """Run the Capo's daily loop.

    Args:
        as_of_date: Trading date
        tickers: Watchlist to analyze. If None, loads from active watchlist.
        mode: "manual" (report to user) or "auto" (execute via Claude)
    """
    print(f"\n{'='*70}")
    print(f"OFFICE DI CAPO — Daily Report")
    print(f"  Date: {as_of_date}")
    print(f"  Mode: {mode}")
    print(f"{'='*70}")

    # 1. Load Capo persona
    print(f"\n  [1] Loading persona...")
    persona = load_persona("Giovanni")
    bookshelf = load_bookshelf(persona)
    print(f"      Giovanni loaded. Bookshelf sections: {list(bookshelf.keys())}")

    # 2. Macro analysis
    print(f"\n  [2] Macro analysis...")
    t0 = time.time()
    macro = run_macro_analysis(as_of_date, persona)
    if macro:
        print(f"      Regime: {macro.regime} (confidence {macro.confidence})")
        print(f"      {macro.reasoning}")
    else:
        print(f"      Macro analysis failed — proceeding with default")
        macro = MacroAnalysis(
            date=as_of_date, regime=Regime.TRANSITIONAL, confidence=3,
            vix_read="unavailable", breadth_read="unavailable",
            credit_read="unavailable", reasoning="Macro analysis failed",
            sectors_favored=[], sectors_avoid=[],
        )
    print(f"      ({time.time()-t0:.0f}s)")

    # 3. Department reports
    print(f"\n  [3] Collecting department reports...")
    if tickers is None:
        tickers = []  # TODO: load from active watchlist
    t0 = time.time()
    reports = collect_department_reports(as_of_date, tickers)
    total_reports = sum(len(v) for v in reports.values())
    print(f"      {total_reports} reports collected ({time.time()-t0:.0f}s)")

    # 4. Portfolio state
    from portfolio import Portfolio
    portfolio = Portfolio(starting_cash=1000.0, log_dir=LEDGER_DIR)
    portfolio_path = os.path.join(LEDGER_DIR, "portfolio.json")
    if os.path.exists(portfolio_path):
        with open(portfolio_path) as f:
            state = json.load(f)
            portfolio.cash = state.get("cash", 1000.0)
            portfolio.positions = state.get("positions", {})
            portfolio.trade_log = state.get("trade_log", [])
    portfolio_state = portfolio.get_state_for_prompt({})

    # 5. Synthesis
    print(f"\n  [4] Capo synthesizing...")
    t0 = time.time()
    report = synthesize(as_of_date, persona, macro, reports, portfolio_state)
    print(f"      ({time.time()-t0:.0f}s)")

    if report:
        print(f"\n  {'='*70}")
        print(f"  CAPO'S REPORT — {as_of_date}")
        print(f"  {'='*70}")
        print(f"  Regime: {report.macro.regime} (conf {report.macro.confidence})")
        print(f"  Rationale: {report.rationale}")
        print(f"  Directives:")
        for d in report.directives:
            print(f"    {d.action.upper():8s} {d.ticker:6s} ${d.amount_usd:.2f} [{d.conviction}] — {d.reason}")
        print(f"  Risk notes: {report.risk_notes}")

        write_ledger(report)

        if mode == "auto":
            print(f"\n  [AUTO] Executing directives...")
            # TODO: execute via Claude or direct portfolio operations
        else:
            print(f"\n  [MANUAL] Report submitted for review.")

    return report


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-03-10"
    mode = sys.argv[2] if len(sys.argv) > 2 else "manual"
    run_daily(date, mode=mode)
