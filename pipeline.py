"""
Medici — Step 1 Pipeline

Orchestrator: load data → compute indicators → build context → score via Qwen → pack payload.
"""

import json
import os
import sys
import time
import requests
import pandas as pd

from indicators import compute_all
from registry import get_registry_for_scoring
from context import build_context, load_vix

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "payloads")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

SCORING_SYSTEM_PROMPT = """You are a quantitative trading analyst. Your job is to evaluate technical indicators for an end-of-day analysis.

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

SCORE_THRESHOLD = 2  # Keep indicators scoring >= this

SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLRE", "XLB", "XLC"]


def load_ticker(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file for {ticker}: {path}")
    return pd.read_parquet(path)


def load_optional(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def load_universe() -> dict:
    """Load all S&P 500 tickers for breadth calculation."""
    ticker_file = os.path.join(os.path.dirname(__file__), "data", "sp500_tickers.txt")
    if not os.path.exists(ticker_file):
        return {}
    with open(ticker_file) as f:
        tickers = [t.strip() for t in f.readlines() if t.strip()]
    universe = {}
    for t in tickers:
        df = load_optional(t)
        if df is not None:
            universe[t] = df
    return universe


def load_sectors() -> dict:
    """Load sector ETF dataframes."""
    sectors = {}
    for s in SECTOR_ETFS:
        df = load_optional(s)
        if df is not None:
            sectors[s] = df
    return sectors


def _chunk_registry(registry: list, max_per_chunk: int = 25) -> list[list]:
    """Split registry into chunks grouped by category, respecting max size."""
    from collections import OrderedDict
    by_cat = OrderedDict()
    for ind in registry:
        cat = ind["category"]
        by_cat.setdefault(cat, []).append(ind)

    chunks = []
    current = []
    for cat, inds in by_cat.items():
        if len(current) + len(inds) > max_per_chunk and current:
            chunks.append(current)
            current = []
        current.extend(inds)
    if current:
        chunks.append(current)
    return chunks


def _score_chunk(chunk: list, market_context: dict) -> dict:
    """Score a single chunk of indicators via Qwen."""
    user_msg = f"""Market context:
{json.dumps(market_context, indent=2)}

Indicator tools to evaluate:
{json.dumps(chunk, indent=2)}

Score each indicator, identify redundancies, and recommend a non-redundant subset."""

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    for attempt in range(3):
        try:
            resp = requests.post(QWEN_API, json=payload, timeout=300)
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"]
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])

            return json.loads(cleaned.strip())
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"      _score_chunk attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(5)

    # Fallback: recommend all indicators in the chunk
    print(f"      _score_chunk giving up, keeping all {len(chunk)} indicators")
    return {
        "scores": [{"name": ind["name"], "score": 4, "reason": "fallback"} for ind in chunk],
        "redundancy_groups": [],
        "recommended_subset": [ind["name"] for ind in chunk],
    }


def score_indicators(market_context: dict) -> dict:
    """Map-reduce scoring: chunk the registry, score each chunk, merge results."""
    registry = get_registry_for_scoring()
    chunks = _chunk_registry(registry)

    all_scores = []
    all_groups = []
    all_recommended = []

    for i, chunk in enumerate(chunks):
        cats = sorted(set(ind["category"] for ind in chunk))
        print(f"      Chunk {i+1}/{len(chunks)}: {len(chunk)} indicators [{', '.join(cats)}]")
        result = _score_chunk(chunk, market_context)
        all_scores.extend(result.get("scores", []))
        all_groups.extend(result.get("redundancy_groups", []))
        all_recommended.extend(result.get("recommended_subset", []))

    return {
        "scores": all_scores,
        "redundancy_groups": all_groups,
        "recommended_subset": all_recommended,
    }


def pack_payload(ticker: str, date: str, market_context: dict,
                 all_indicators: dict, scoring_result: dict) -> dict:
    """Pack only the surviving indicators into the final payload."""
    
    recommended = set(scoring_result.get("recommended_subset", []))

    # Fallback: Only use raw scores if Qwen failed to provide a subset
    if not recommended:
        for s in scoring_result.get("scores", []):
            if s.get("score", 0) >= 4:  # Only keep 'High value' or 'Critical'
                recommended.add(s["name"])

    # Hard cap the payload size to prevent VRAM overflow / attention collapse
    MAX_INDICATORS = 15
    recommended = set(list(recommended)[:MAX_INDICATORS])

    filtered = {k: v for k, v in all_indicators.items() if k in recommended}

    return {
        "ticker": ticker,
        "date": date,
        "context": market_context,
        "scoring_summary": {
            "total_evaluated": len(scoring_result.get("scores", [])),
            "kept": len(filtered),
            "redundancy_groups": scoring_result.get("redundancy_groups", []),
        },
        "indicators": filtered,
    }


def run(ticker: str):
    """Run the full Step 1 pipeline for a ticker."""
    print(f"\n{'='*60}")
    print(f"MEDICI STEP 1 — {ticker}")
    print(f"{'='*60}")

    # 1. Load data
    print(f"\n[1] Loading data...")
    df = load_ticker(ticker)
    vix_df = load_vix()
    spy_df = load_optional("SPY") if ticker != "SPY" else df
    hyg_df = load_optional("HYG")
    tlt_df = load_optional("TLT")
    print(f"    {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")

    print(f"    Loading universe for breadth...")
    t0 = time.time()
    universe = load_universe()
    sectors = load_sectors()
    print(f"    {len(universe)} tickers, {len(sectors)} sectors loaded in {time.time()-t0:.1f}s")

    # 2. Compute indicators
    print(f"[2] Computing indicators...")
    t0 = time.time()
    all_indicators = compute_all(
        df, spy_df=spy_df, universe_dfs=universe, sector_dfs=sectors,
        hyg_df=hyg_df, tlt_df=tlt_df,
    )
    elapsed = time.time() - t0
    computed = sum(1 for v in all_indicators.values() if v is not None)
    print(f"    {computed}/{len(all_indicators)} indicators computed in {elapsed:.2f}s")

    # 3. Build context
    print(f"[3] Building market context...")
    ctx = build_context(ticker, df, vix_df)
    print(f"    Regime: {ctx['regime']}")

    # 4. Score via Qwen
    print(f"[4] Scoring indicators via Qwen...")
    t0 = time.time()
    scoring = score_indicators(ctx)
    elapsed = time.time() - t0
    print(f"    Scored in {elapsed:.1f}s")
    print(f"    Recommended subset: {scoring.get('recommended_subset', [])}")

    # 5. Pack payload
    print(f"[5] Packing payload...")
    payload = pack_payload(ticker, ctx["date"], ctx, all_indicators, scoring)
    kept = payload["scoring_summary"]["kept"]
    total = payload["scoring_summary"]["total_evaluated"]
    print(f"    Kept {kept}/{total} indicators")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_{ctx['date']}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n    Payload saved: {out_path}")

    return payload


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    run(ticker)
