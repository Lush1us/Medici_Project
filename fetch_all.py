"""
Medici — Full Universe Data Fetcher

Pulls 2y EOD data for all S&P 500 components + macro instruments.
"""

import pandas as pd
import yfinance as yf
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")
TICKER_FILE = os.path.join(os.path.dirname(__file__), "data", "sp500_tickers.txt")

MACRO = [
    ("^VIX", "VIX"),
    ("TLT", "TLT"),
    ("GLD", "GLD"),
    ("USO", "USO"),
    ("UUP", "UUP"),     # US Dollar
    ("HYG", "HYG"),     # High Yield Corp Bonds
    ("IEF", "IEF"),     # 7-10 Year Treasury
    ("XLF", "XLF"),     # Financials sector
    ("XLE", "XLE"),     # Energy sector
    ("XLK", "XLK"),     # Tech sector
    ("XLV", "XLV"),     # Healthcare sector
    ("XLI", "XLI"),     # Industrials sector
    ("XLP", "XLP"),     # Consumer Staples
    ("XLY", "XLY"),     # Consumer Discretionary
    ("XLU", "XLU"),     # Utilities
    ("XLRE", "XLRE"),   # Real Estate
    ("XLB", "XLB"),     # Materials
    ("XLC", "XLC"),     # Communication Services
    ("IWM", "IWM"),     # Russell 2000
    ("DIA", "DIA"),     # Dow 30
    ("QQQ", "QQQ"),     # Nasdaq 100
]

PERIOD = "2y"
WORKERS = 8


def fetch_one(yf_ticker, save_as):
    """Fetch a single ticker. Returns (save_as, rows, start, end) or (save_as, None, error)."""
    out_path = os.path.join(DATA_DIR, f"{save_as}.parquet")
    try:
        df = yf.download(yf_ticker, period=PERIOD, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return (save_as, 0, "no data")
        # yfinance returns MultiIndex columns (Price, Ticker) for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(level=1, axis=1)
        # Deduplicate any remaining duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        df.to_parquet(out_path)
        return (save_as, len(df), None)
    except Exception as e:
        return (save_as, 0, str(e))


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load S&P 500 tickers
    with open(TICKER_FILE) as f:
        sp500 = [(t.strip(), t.strip()) for t in f.readlines() if t.strip()]

    # Combine: macro + sp500, deduplicate by save_as
    seen = set()
    all_tickers = []
    for yf_t, save_as in MACRO + sp500:
        if save_as not in seen:
            all_tickers.append((yf_t, save_as))
            seen.add(save_as)

    total = len(all_tickers)
    print(f"Fetching {total} tickers ({PERIOD} daily) with {WORKERS} threads")
    print(f"Output: {DATA_DIR}/\n")

    success = 0
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_one, yf_t, save_as): save_as for yf_t, save_as in all_tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            save_as, rows, err = future.result()
            if err:
                failed.append((save_as, err))
            else:
                success += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed — {success} ok, {len(failed)} failed")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. {success}/{total} tickers saved.")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed[:20]:
            print(f"  {name}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Update manifest
    manifest = {
        "fetched_at": datetime.now().isoformat(),
        "period": PERIOD,
        "total": total,
        "success": success,
        "failed": len(failed),
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
