"""
Department of Technical Analysis — Ticker Deep Dive

Extended analysis for a single ticker when requested by the Capo.
Same as daily report but with more detail.
"""

from Headquarters.Dpt_of_Technical_Analysis.technical_report import compute_technical


def run_ticker(ticker: str, as_of_date: str) -> dict | None:
    """Deep dive technical analysis — same compute, flagged as deep dive."""
    result = compute_technical(ticker, as_of_date)
    if result:
        result["report_type"] = "deep_dive"
    return result
