"""
Medici — Batch pipeline test across multiple tickers.
Runs Step 1 + agent analysis for a sample to validate the chain holds.
"""

import json
import time
from pipeline import run as run_pipeline
from agent import run_agent

TICKERS = ["SPY", "NVDA", "AAPL", "XLE", "TSLA"]


def main():
    print("MEDICI — BATCH PIPELINE TEST")
    print(f"Tickers: {TICKERS}\n")

    results = {}
    t0 = time.time()

    for ticker in TICKERS:
        try:
            # Step 1: pipeline
            payload = run_pipeline(ticker)

            # Step 2: agent analysis
            thesis = run_agent("average_joe", ticker)

            results[ticker] = {
                "pipeline": "OK",
                "indicators_kept": payload["scoring_summary"]["kept"],
                "agent": "OK" if thesis else "FAILED",
                "bias": thesis.get("bias") if thesis else None,
                "would_trade": thesis.get("would_trade") if thesis else None,
            }
        except Exception as e:
            results[ticker] = {"pipeline": "FAILED", "error": str(e)}

    elapsed = time.time() - t0

    print(f"\n\n{'='*60}")
    print(f"BATCH RESULTS ({elapsed:.0f}s total)")
    print(f"{'='*60}")
    for ticker, r in results.items():
        if r.get("error"):
            print(f"  {ticker:6s}: FAILED — {r['error']}")
        else:
            print(f"  {ticker:6s}: {r['indicators_kept']} indicators | {r.get('bias', '?'):8s} | trade: {r.get('would_trade', '?')}")


if __name__ == "__main__":
    main()
