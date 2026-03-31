"""
Medici — Full US Market Data Fetcher

Pulls 2y EOD data for all US-listed stocks and ETFs.
"""

import yfinance as yf
import pandas as pd
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")
BASE_DIR = os.path.dirname(__file__)

PERIOD = "2y"
WORKERS = 16


def load_tickers():
    """Load all tickers from stock and ETF lists, deduplicate against what we already have."""
    tickers = set()

    for f in ["us_stocks.txt", "us_etfs.txt", "sp500_tickers.txt"]:
        path = os.path.join(BASE_DIR, "data", f)
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    t = line.strip()
                    if t:
                        tickers.add(t)

    # Add macro instruments with special yfinance symbols
    macro = {"^VIX": "VIX", "^VIX3M": "VIX3M", "^VIX9D": "VIX9D"}
    return tickers, macro


def already_fetched():
    """Return set of tickers we already have parquet files for."""
    if not os.path.exists(DATA_DIR):
        return set()
    return {f.replace(".parquet", "") for f in os.listdir(DATA_DIR) if f.endswith(".parquet")}


def fetch_one(yf_ticker, save_as):
    out_path = os.path.join(DATA_DIR, f"{save_as}.parquet")
    try:
        df = yf.download(yf_ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return (save_as, 0, "no data")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(out_path)
        return (save_as, len(df), None)
    except Exception as e:
        return (save_as, 0, str(e)[:80])


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tickers, macro = load_tickers()
    existing = already_fetched()

    # Build fetch list: skip already downloaded
    to_fetch = []
    for yf_sym, save_as in macro.items():
        if save_as not in existing:
            to_fetch.append((yf_sym, save_as))
    for t in sorted(tickers):
        if t not in existing:
            to_fetch.append((t, t))

    total = len(to_fetch)
    already = len(existing)
    print(f"Universe: {len(tickers) + len(macro)} tickers")
    print(f"Already fetched: {already}")
    print(f"Remaining: {total}")
    print(f"Workers: {WORKERS}")

    if total == 0:
        print("Nothing to fetch.")
        return

    print(f"\nFetching...\n")

    success = 0
    failed = 0
    no_data = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, yf_t, save_as): save_as for yf_t, save_as in to_fetch}
        done = 0
        for future in as_completed(futures):
            done += 1
            save_as, rows, err = future.result()
            if err == "no data":
                no_data += 1
            elif err:
                failed += 1
            else:
                success += 1

            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.0f}s | {success} ok, {failed} err, {no_data} empty | ~{eta:.0f}s remaining")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"  New: {success}")
    print(f"  Failed: {failed}")
    print(f"  No data: {no_data}")
    print(f"  Total on disk: {already + success}")

    # Update manifest
    manifest = {
        "fetched_at": datetime.now().isoformat(),
        "period": PERIOD,
        "on_disk": already + success,
        "new_this_run": success,
        "failed": failed,
        "no_data": no_data,
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
