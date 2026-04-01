"""
Department of Sentiment Analysis — Ticker Deep Dive
"""

from Headquarters.Dpt_of_Sentiment_Analysis.sentiment_report import compute_sentiment


def run_ticker(ticker: str, as_of_date: str) -> dict | None:
    result = compute_sentiment(ticker, as_of_date)
    if result:
        result["report_type"] = "deep_dive"
    return result
