"""
Quick test: run Piero, Cosimo, Caterina pitches on 30 tickers to validate fix.
"""

import json
import time
from weekend_pipeline import hard_filter, run_giovanni_macro, run_solo_pitch

TEST_AGENTS = ["Piero", "Cosimo", "Caterina"]
SAMPLE_SIZE = 30

date = "2025-03-07"

print(f"Hard filter...")
universe = hard_filter(date)
print(f"  {len(universe)} passed")

print(f"\nGiovanni macro...")
macro = run_giovanni_macro(date)
if macro:
    print(f"  {macro.get('regime')} (conf {macro.get('confidence')})")

# Take first 30
sample = universe[:SAMPLE_SIZE]
print(f"\nTesting {len(sample)} tickers × {len(TEST_AGENTS)} agents\n")

results = {}  # {agent: [pitched_tickers]}
for agent in TEST_AGENTS:
    results[agent] = []

for i, td in enumerate(sample):
    ticker = td["ticker"]
    for agent in TEST_AGENTS:
        r = run_solo_pitch(agent, td, macro)
        if r and r.get("pitch"):
            results[agent].append(ticker)
            print(f"  {agent:10s} → {ticker:6s} YES: {r.get('reason','')[:60]}")

    if (i + 1) % 10 == 0:
        print(f"  --- {i+1}/{SAMPLE_SIZE} ---")

print(f"\n{'='*50}")
print(f"RESULTS (out of {SAMPLE_SIZE} tickers):")
for agent in TEST_AGENTS:
    picks = results[agent]
    pct = len(picks) / SAMPLE_SIZE * 100
    print(f"  {agent:10s}: {len(picks):2d} picks ({pct:.0f}%) — {picks}")
print(f"{'='*50}")
