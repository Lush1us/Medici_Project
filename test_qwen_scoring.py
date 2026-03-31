"""
Medici — Qwen 4B Indicator Scoring Test

Tests whether Qwen 3.5 4B can:
1. Parse a list of technical indicators with metadata
2. Score them for relevance given a market context
3. Identify redundancies between correlated indicators
4. Produce structured JSON output

We escalate difficulty across 3 rounds to find the limits.
"""

import json
import time
import requests
import sys

API = "http://localhost:8080/v1/chat/completions"
MODEL = "Qwen3.5-4B-Q6_K.gguf"

# --- Test Data ---

# Small set (Round 1): 8 indicators, obvious redundancies
INDICATORS_SMALL = [
    {"name": "RSI_14", "category": "momentum", "description": "Relative Strength Index, 14-period. Measures overbought/oversold on 0-100 scale."},
    {"name": "RSI_7", "category": "momentum", "description": "Relative Strength Index, 7-period. Faster RSI variant."},
    {"name": "MACD_12_26_9", "category": "trend", "description": "Moving Average Convergence Divergence. Signal line crossovers indicate trend changes."},
    {"name": "EMA_12", "category": "trend", "description": "12-period Exponential Moving Average. Component of MACD."},
    {"name": "EMA_26", "category": "trend", "description": "26-period Exponential Moving Average. Component of MACD."},
    {"name": "ATR_14", "category": "volatility", "description": "Average True Range, 14-period. Measures price volatility."},
    {"name": "BBANDS_20_2", "category": "volatility", "description": "Bollinger Bands, 20-period, 2 std dev. Volatility envelope around SMA."},
    {"name": "OBV", "category": "volume", "description": "On-Balance Volume. Cumulative volume flow confirming price trends."},
]

# Medium set (Round 2): 16 indicators, subtler overlaps
INDICATORS_MEDIUM = INDICATORS_SMALL + [
    {"name": "STOCH_14_3_3", "category": "momentum", "description": "Stochastic Oscillator. %K and %D lines measure momentum, similar range to RSI."},
    {"name": "WILLR_14", "category": "momentum", "description": "Williams %R. Inverted stochastic, measures overbought/oversold on -100 to 0."},
    {"name": "ADX_14", "category": "trend", "description": "Average Directional Index. Measures trend strength 0-100, not direction."},
    {"name": "SMA_50", "category": "trend", "description": "50-period Simple Moving Average. Medium-term trend reference."},
    {"name": "SMA_200", "category": "trend", "description": "200-period Simple Moving Average. Long-term trend reference."},
    {"name": "VWAP", "category": "volume", "description": "Volume-Weighted Average Price. Institutional fair value benchmark."},
    {"name": "CCI_20", "category": "momentum", "description": "Commodity Channel Index. Measures deviation from statistical mean."},
    {"name": "MFI_14", "category": "volume", "description": "Money Flow Index. Volume-weighted RSI variant."},
]

# Large set (Round 3): 24 indicators, needs real judgment about which combos matter
INDICATORS_LARGE = INDICATORS_MEDIUM + [
    {"name": "ICHIMOKU", "category": "trend", "description": "Ichimoku Cloud. Multi-component trend system: Tenkan, Kijun, Senkou A/B, Chikou."},
    {"name": "PSAR", "category": "trend", "description": "Parabolic SAR. Trailing stop-and-reverse dots above/below price."},
    {"name": "KELTNER_20_1.5", "category": "volatility", "description": "Keltner Channels. ATR-based envelope around EMA, similar to Bollinger but uses ATR."},
    {"name": "TRIX_15", "category": "momentum", "description": "Triple Exponential Average. Rate of change of triple-smoothed EMA, filters noise."},
    {"name": "CMF_20", "category": "volume", "description": "Chaikin Money Flow. Accumulation/distribution over 20 periods."},
    {"name": "ROC_12", "category": "momentum", "description": "Rate of Change, 12-period. Simple percentage price change."},
    {"name": "AROON_25", "category": "trend", "description": "Aroon indicator. Measures time since highest high / lowest low."},
    {"name": "STDDEV_20", "category": "volatility", "description": "Standard Deviation of price, 20-period. Raw volatility measure, component of Bollinger."},
]

MARKET_CONTEXT = {
    "ticker": "SPY",
    "date": "2026-03-27",
    "close": 542.18,
    "change_pct": -1.23,
    "volume_vs_avg": 1.45,
    "regime": "Downtrend with elevated volatility. Price below 50-SMA, above 200-SMA. VIX at 22.4.",
}

SYSTEM_PROMPT = """You are a quantitative trading analyst. Your job is to evaluate technical indicators for an end-of-day analysis.

For each indicator, assign a score 1-5:
5 = Critical for current regime
4 = High value
3 = Useful context
2 = Low marginal value (redundant or less relevant)
1 = Skip (fully redundant or irrelevant)

You MUST also identify redundancy groups: indicators that measure essentially the same thing.

Respond in valid JSON only. No markdown, no explanation outside JSON. Format:
{
  "scores": [{"name": "INDICATOR", "score": N, "reason": "brief"}],
  "redundancy_groups": [{"group": "label", "members": ["IND1", "IND2"], "keep": "IND1", "why": "brief"}],
  "recommended_subset": ["IND1", "IND2", ...]
}"""


def run_test(indicators, label, ctx_tokens=4096):
    user_msg = f"""Market context:
{json.dumps(MARKET_CONTEXT, indent=2)}

Indicator tools to evaluate:
{json.dumps(indicators, indent=2)}

Score each indicator, identify redundancies, and recommend a non-redundant subset."""

    print(f"\n{'='*60}")
    print(f"ROUND: {label} ({len(indicators)} indicators)")
    print(f"{'='*60}")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": ctx_tokens,
    }

    t0 = time.time()
    try:
        resp = requests.post(API, json=payload, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        print(f"  REQUEST FAILED: {e}")
        return None

    elapsed = time.time() - t0
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    tokens_in = data.get("usage", {}).get("prompt_tokens", "?")
    tokens_out = data.get("usage", {}).get("completion_tokens", "?")

    print(f"  Time: {elapsed:.1f}s | Tokens in: {tokens_in} | Tokens out: {tokens_out}")

    # Try to parse JSON
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        valid_json = True
    except json.JSONDecodeError as e:
        print(f"  JSON PARSE FAILED: {e}")
        print(f"  Raw output (first 500 chars):\n{raw[:500]}")
        return {"valid_json": False, "raw": raw, "elapsed": elapsed}

    print(f"  Valid JSON: YES")

    # --- Quality checks ---
    issues = []

    # Check all indicators scored
    scored_names = {s["name"] for s in result.get("scores", [])}
    input_names = {i["name"] for i in indicators}
    missing = input_names - scored_names
    extra = scored_names - input_names
    if missing:
        issues.append(f"Missing scores for: {missing}")
    if extra:
        issues.append(f"Hallucinated indicators: {extra}")

    # Check score range
    bad_scores = [s for s in result.get("scores", []) if s.get("score", 0) not in range(1, 6)]
    if bad_scores:
        issues.append(f"Out-of-range scores: {bad_scores}")

    # Check redundancy sanity
    groups = result.get("redundancy_groups", [])
    known_redundancies = {
        frozenset({"RSI_14", "RSI_7"}),
        frozenset({"EMA_12", "EMA_26", "MACD_12_26_9"}),
    }
    found_groups = [frozenset(g["members"]) for g in groups]

    for kr in known_redundancies:
        if kr.issubset(input_names):
            matched = any(kr.issubset(fg) for fg in found_groups)
            if not matched:
                # Check if at least partially caught
                partial = any(len(kr & fg) > 1 for fg in found_groups)
                if partial:
                    issues.append(f"Partially caught redundancy: {set(kr)}")
                else:
                    issues.append(f"Missed obvious redundancy: {set(kr)}")

    # Check recommended subset
    subset = result.get("recommended_subset", [])
    if not subset:
        issues.append("No recommended subset provided")
    elif len(subset) >= len(indicators):
        issues.append("Subset is same size as input — didn't filter anything")

    # Check if subset only contains scored indicators
    subset_set = set(subset)
    if subset_set - input_names:
        issues.append(f"Subset contains unknown indicators: {subset_set - input_names}")

    if issues:
        print(f"  Issues ({len(issues)}):")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"  Quality: ALL CHECKS PASSED")

    # Print summary
    print(f"\n  Scores:")
    for s in sorted(result.get("scores", []), key=lambda x: x.get("score", 0), reverse=True):
        print(f"    {s['name']:20s} -> {s.get('score', '?')}  ({s.get('reason', '')})")

    print(f"\n  Redundancy groups:")
    for g in groups:
        print(f"    {g.get('group', '?')}: {g.get('members', [])} -> keep {g.get('keep', '?')}")

    print(f"\n  Recommended subset ({len(subset)}): {subset}")

    return {
        "valid_json": True,
        "issues": issues,
        "n_indicators": len(indicators),
        "n_subset": len(subset),
        "elapsed": elapsed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "result": result,
    }


def main():
    print("Medici — Qwen 4B Indicator Scoring Capability Test")
    print("=" * 60)

    results = {}

    results["small"] = run_test(INDICATORS_SMALL, "SMALL (8 indicators)", ctx_tokens=2048)
    results["medium"] = run_test(INDICATORS_MEDIUM, "MEDIUM (16 indicators)", ctx_tokens=3072)
    results["large"] = run_test(INDICATORS_LARGE, "LARGE (24 indicators)", ctx_tokens=4096)

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for label, r in results.items():
        if r is None:
            print(f"  {label:8s}: FAILED (request error)")
        elif not r.get("valid_json"):
            print(f"  {label:8s}: FAILED (invalid JSON)")
        else:
            n_issues = len(r.get("issues", []))
            status = "CLEAN" if n_issues == 0 else f"{n_issues} ISSUES"
            print(f"  {label:8s}: {status} | {r['n_indicators']} -> {r['n_subset']} indicators | {r['elapsed']:.1f}s | {r['tokens_in']}/{r['tokens_out']} tokens")


if __name__ == "__main__":
    main()
