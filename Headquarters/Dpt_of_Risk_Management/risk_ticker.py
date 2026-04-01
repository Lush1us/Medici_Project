"""
Department of Risk Management — Ticker Deep Dive
"""

from Headquarters.Dpt_of_Risk_Management.risk_report import compute_risk


def run_ticker(ticker: str, as_of_date: str) -> dict | None:
    result = compute_risk(ticker, as_of_date)
    if result:
        result["report_type"] = "deep_dive"
    return result
