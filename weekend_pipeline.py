"""
Medici — Weekend Pipeline (Watchlist Builder)

Runs asynchronously over the weekend using local Qwen.
4 steps: Hard Filter → Solo Pitch → Targeted Q&A → Vote → Watchlist

Usage:
    python weekend_pipeline.py [as_of_date]
    python weekend_pipeline.py 2025-03-07
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "eod")
WATCHLIST_DIR = os.path.join(os.path.dirname(__file__), "data", "watchlists")
CONSIGLIO_DIR = os.path.join(os.path.dirname(__file__), "Consiglio")

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

PERSONAS = ["Cosimo", "Lorenzo", "Giuliano", "Giovanni", "Caterina", "Piero"]

# Giovanni runs separately on macro — he doesn't solo-pitch individual stocks
STOCK_PICKERS = ["Cosimo", "Lorenzo", "Giuliano", "Caterina", "Piero"]


# =============================================================================
# HELPERS
# =============================================================================

def _load_ticker(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_persona(name: str) -> str:
    path = os.path.join(CONSIGLIO_DIR, name, "brain.md")
    with open(path) as f:
        return f.read()


def _condensed_persona(name: str) -> str:
    """Extract Identity + Trading Style + Indicator Preferences only."""
    text = _load_persona(name)
    lines = text.split("\n")
    keep = []
    in_section = False
    for line in lines:
        if line.startswith("## Identity") or line.startswith("## Trading Style") or line.startswith("## Indicator Preferences"):
            in_section = True
            keep.append(line)
        elif line.startswith("## ") and in_section:
            in_section = False
        elif in_section:
            keep.append(line)
    return "\n".join(keep)


def _qwen_call(system: str, user_msg: str, label: str = "") -> dict | None:
    try:
        resp = requests.post(QWEN_API, json={
            "model": QWEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        return json.loads(cleaned.strip())
    except Exception as e:
        print(f"      {label} FAILED: {e}")
        return None


def _load_sp500_tickers() -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "data", "sp500_tickers.txt")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [t.strip() for t in f if t.strip()]


# =============================================================================
# STEP 1: HARD FILTER (pure Python)
# =============================================================================

def hard_filter(as_of_date: str) -> list[dict]:
    """Filter universe for liquid, tradeable stocks.
    Rules: Close > $5, 20d avg volume > 1M, 20d avg dollar volume > $10M.
    """
    tickers = _load_sp500_tickers()
    survivors = []

    for ticker in tickers:
        df = _load_ticker(ticker)
        if df is None:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 50:
            continue

        close = float(df["Close"].iloc[-1])
        if close <= 5.0:
            continue

        avg_vol = float(df["Volume"].tail(20).mean())
        if avg_vol < 1_000_000:
            continue

        avg_dollar_vol = float((df["Close"] * df["Volume"]).tail(20).mean())
        if avg_dollar_vol < 10_000_000:
            continue

        # Compute technicals for pitch context
        import ta as ta_lib
        import numpy as np

        rsi = float(ta_lib.momentum.rsi(df["Close"], window=14).iloc[-1])
        adx_obj = ta_lib.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        adx = float(adx_obj.adx().iloc[-1])
        sma_50 = float(df["Close"].rolling(50).mean().iloc[-1])
        sma_200 = float(df["Close"].rolling(200).mean().iloc[-1]) if len(df) >= 200 else None
        rvol = round(float(df["Volume"].iloc[-1]) / avg_vol, 2)

        # Extra metrics for Cosimo (trend + volume confluence)
        macd_obj = ta_lib.trend.MACD(df["Close"])
        macd_hist = float(macd_obj.macd_diff().iloc[-1]) if not pd.isna(macd_obj.macd_diff().iloc[-1]) else 0
        cmf = float(ta_lib.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"]).iloc[-1]) if len(df) >= 20 else 0
        obv_slope = 0.0
        obv = ta_lib.volume.on_balance_volume(df["Close"], df["Volume"])
        if len(obv) >= 10:
            obv_slope = round(float(obv.iloc[-1] - obv.iloc[-10]) / (abs(float(obv.iloc[-10])) + 1) * 100, 2)

        # Extra metrics for Caterina (microstructure/liquidity)
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        dollar_vol = df["Close"] * df["Volume"]
        amihud_raw = (log_ret.abs() / dollar_vol).tail(20).mean()
        amihud = round(float(amihud_raw * 1e6), 4) if not pd.isna(amihud_raw) else None
        # CLV
        hl_range = df["High"].iloc[-1] - df["Low"].iloc[-1]
        clv = round(((df["Close"].iloc[-1] - df["Low"].iloc[-1]) - (df["High"].iloc[-1] - df["Close"].iloc[-1])) / hl_range, 2) if hl_range > 0 else 0
        # Bar range ratio
        bar_range = float(df["High"].iloc[-1] - df["Low"].iloc[-1])
        avg_bar_range = float((df["High"] - df["Low"]).tail(20).mean())
        range_ratio = round(bar_range / avg_bar_range, 2) if avg_bar_range > 0 else 1.0

        # Extra for Piero (mean-reversion specifics)
        std_50 = float(df["Close"].rolling(50).std().iloc[-1]) if len(df) >= 50 else None
        zscore_50 = round((close - sma_50) / std_50, 2) if std_50 and std_50 > 0 else None

        survivors.append({
            "ticker": ticker,
            "close": round(close, 2),
            "adx": round(adx, 1),
            "rsi": round(rsi, 1),
            "rvol": rvol,
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "pct_from_200sma": round((close / sma_200 - 1) * 100, 1) if sma_200 else None,
            "macd_hist": round(macd_hist, 3),
            "cmf": round(cmf, 3),
            "obv_slope_pct": obv_slope,
            "amihud": amihud,
            "clv": clv,
            "range_ratio": range_ratio,
            "zscore_50": zscore_50,
        })

    return survivors


# =============================================================================
# STEP 1.5: GIOVANNI'S MACRO READ
# =============================================================================

MACRO_SYSTEM = """You are Giovanni, a macro specialist. Your profile:
{persona}

Evaluate the current macro environment and produce a regime call.

Respond in valid JSON only. No markdown. Format:
{{
  "regime": "risk-on" | "risk-off" | "transitional",
  "confidence": 1-10,
  "reasoning": "2-3 sentences on what the macro signals say",
  "bias_for_longs": "favorable" | "unfavorable" | "neutral",
  "sectors_favored": ["sector ETFs that look strong"],
  "sectors_avoid": ["sector ETFs showing weakness"]
}}"""


def run_giovanni_macro(as_of_date: str) -> dict | None:
    """Giovanni reads macro instruments and produces a regime call."""
    persona_text = _condensed_persona("Giovanni")

    # Gather macro data
    macro = {}
    for ticker in ["SPY", "HYG", "TLT", "VIX"]:
        df = _load_ticker(ticker)
        if df is None:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 50:
            continue
        close = float(df["Close"].iloc[-1])
        sma_50 = float(df["Close"].rolling(50).mean().iloc[-1])
        change_5d = round(float(df["Close"].pct_change(5).iloc[-1]) * 100, 2)
        change_20d = round(float(df["Close"].pct_change(20).iloc[-1]) * 100, 2)
        macro[ticker] = {"close": round(close, 2), "sma_50": round(sma_50, 2),
                         "5d_change": change_5d, "20d_change": change_20d}

    # Breadth
    tickers = _load_sp500_tickers()
    above_200 = 0
    total = 0
    for t in tickers:
        df = _load_ticker(t)
        if df is None or len(df) < 200:
            continue
        df = df.loc[:as_of_date]
        if len(df) < 200:
            continue
        total += 1
        if float(df["Close"].iloc[-1]) > float(df["Close"].rolling(200).mean().iloc[-1]):
            above_200 += 1
    if total > 0:
        macro["breadth_pct_above_200sma"] = round(above_200 / total * 100, 1)

    # Credit spread proxy
    if "HYG" in macro and "TLT" in macro:
        macro["credit_note"] = "HYG outperforming TLT = risk-on, underperforming = risk-off"

    system = MACRO_SYSTEM.format(persona=persona_text)
    user_msg = f"""Date: {as_of_date}\n\nMacro data:\n{json.dumps(macro)}"""

    return _qwen_call(system, user_msg, "Giovanni macro")


# =============================================================================
# STEP 2: SOLO PITCH (one ticker at a time per agent)
# =============================================================================

PITCH_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

You are scanning for stocks to add to your weekly watchlist. You will see one stock's metrics. If the data hints at a setup you'd want to explore deeper, pitch it. You are NOT committing to a trade — just flagging for a closer look.

Be selective. A single metric being extreme is not enough — you need at least two signals that together suggest something worth investigating. Pass on anything borderline.

Respond in valid JSON only. No markdown. Format:
{{
  "pitch": true | false,
  "reason": "one sentence why this caught your eye or why you're passing"
}}"""


def run_solo_pitch(persona_name: str, ticker_data: dict,
                   macro_regime: dict | None) -> dict | None:
    persona_text = _condensed_persona(persona_name)
    system = PITCH_SYSTEM.format(name=persona_name, persona=persona_text)

    macro_line = ""
    if macro_regime:
        macro_line = f"\nMacro: {macro_regime.get('regime', '?')}"

    user_msg = f"""{ticker_data['ticker']}: Close=${ticker_data['close']} ADX={ticker_data['adx']} RSI={ticker_data['rsi']} RVOL={ticker_data['rvol']} SMA50={ticker_data['sma_50']} SMA200={ticker_data.get('sma_200','N/A')} Pct200={ticker_data.get('pct_from_200sma','N/A')}% MACD_H={ticker_data.get('macd_hist','N/A')} CMF={ticker_data.get('cmf','N/A')} OBV%={ticker_data.get('obv_slope_pct','N/A')} Amihud={ticker_data.get('amihud','N/A')} CLV={ticker_data.get('clv','N/A')} RangeR={ticker_data.get('range_ratio','N/A')} Z50={ticker_data.get('zscore_50','N/A')}{macro_line}

Interesting enough for your watchlist?"""

    return _qwen_call(system, user_msg, f"{persona_name}→{ticker_data['ticker']}")


# =============================================================================
# STEP 3: TARGETED Q&A
# =============================================================================

QNA_QUESTION_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

Another advisor pitched {ticker} as a trade. You DISAGREED. Ask ONE sharp question that challenges their thesis. Be specific — reference the data.

Respond in valid JSON only. No markdown. Format:
{{
  "question": "your single question"
}}"""

QNA_ANSWER_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

You pitched {ticker}. Another advisor has a question challenging your thesis. Answer concisely.

Respond in valid JSON only. No markdown. Format:
{{
  "answer": "your response in 2-3 sentences max"
}}"""


def run_qna(ticker: str, pitcher: str, pitch_reason: str,
            dissenter: str, ticker_data: dict) -> dict | None:
    """One dissenter asks one question, pitcher answers."""
    # Question
    q_persona = _condensed_persona(dissenter)
    q_system = QNA_QUESTION_SYSTEM.format(name=dissenter, persona=q_persona, ticker=ticker)
    q_msg = f"""{pitcher} pitched {ticker}: "{pitch_reason}"
Data: Close=${ticker_data['close']} ADX={ticker_data['adx']} RSI={ticker_data['rsi']} RVOL={ticker_data['rvol']}

Ask your question."""

    q_result = _qwen_call(q_system, q_msg, f"{dissenter}?→{pitcher}({ticker})")
    if not q_result:
        return None

    question = q_result.get("question", "")

    # Answer
    a_persona = _condensed_persona(pitcher)
    a_system = QNA_ANSWER_SYSTEM.format(name=pitcher, persona=a_persona, ticker=ticker)
    a_msg = f"""You pitched {ticker}: "{pitch_reason}"
{dissenter} asks: "{question}"

Respond."""

    a_result = _qwen_call(a_system, a_msg, f"{pitcher}!→{dissenter}({ticker})")
    if not a_result:
        return {"questioner": dissenter, "question": question, "answer": None}

    return {
        "questioner": dissenter,
        "question": question,
        "answer": a_result.get("answer", ""),
    }


# =============================================================================
# STEP 4: VOTE
# =============================================================================

VOTE_SYSTEM = """You are {name}, a trading analyst. Key traits:
{persona}

You will see a ticker that was pitched by another advisor, along with a Q&A exchange about it. Vote whether this ticker should be on the weekly watchlist.

Respond in valid JSON only. No markdown. Format:
{{
  "vote": "stay" | "go",
  "reason": "one sentence"
}}"""


def run_vote(persona_name: str, ticker: str, pitch_reason: str,
             qna_exchanges: list, ticker_data: dict) -> dict | None:
    persona_text = _condensed_persona(persona_name)
    system = VOTE_SYSTEM.format(name=persona_name, persona=persona_text)

    qna_text = ""
    for ex in qna_exchanges:
        if ex:
            qna_text += f"\n  Q ({ex['questioner']}): {ex['question']}"
            qna_text += f"\n  A: {ex.get('answer', 'no response')}"

    user_msg = f"""{ticker}: Close=${ticker_data['close']} ADX={ticker_data['adx']} RSI={ticker_data['rsi']} RVOL={ticker_data['rvol']}
Pitched by advisor: "{pitch_reason}"
Q&A:{qna_text if qna_text else ' none'}

Vote: should {ticker} stay on the watchlist?"""

    return _qwen_call(system, user_msg, f"{persona_name} vote({ticker})")


# =============================================================================
# MAIN WEEKEND PIPELINE
# =============================================================================

def run_weekend_pipeline(as_of_date: str):
    os.makedirs(WATCHLIST_DIR, exist_ok=True)
    t_start = time.time()

    print(f"\n{'='*70}")
    print(f"MEDICI WEEKEND PIPELINE — Watchlist Builder")
    print(f"  As of: {as_of_date}")
    print(f"  Agents: {', '.join(PERSONAS)}")
    print(f"{'='*70}")

    # =========================================================================
    # STEP 1: HARD FILTER
    # =========================================================================
    print(f"\n  [STEP 1] Hard filter...")
    t0 = time.time()
    universe = hard_filter(as_of_date)
    print(f"           {len(universe)} stocks passed (Close>$5, AvgVol>1M, DolVol>$10M) in {time.time()-t0:.1f}s")

    if not universe:
        print("           No survivors. Aborting.")
        return

    # =========================================================================
    # STEP 1.5: GIOVANNI MACRO READ
    # =========================================================================
    print(f"\n  [STEP 1.5] Giovanni macro read...")
    t0 = time.time()
    macro_regime = run_giovanni_macro(as_of_date)
    if macro_regime:
        print(f"             Regime: {macro_regime.get('regime', '?')} (conf {macro_regime.get('confidence', '?')})")
        print(f"             {macro_regime.get('reasoning', '')}")
    else:
        print(f"             FAILED — proceeding without macro context")
    print(f"             ({time.time()-t0:.0f}s)")

    # =========================================================================
    # STEP 2: SOLO PITCH (one ticker at a time)
    # =========================================================================
    print(f"\n  [STEP 2] Solo pitches ({len(universe)} tickers × {len(STOCK_PICKERS)} agents)...")

    # {ticker: {persona: {"pitch": bool, "reason": str}}}
    all_pitches = {}
    pitched_tickers = {}  # {ticker: [(persona, reason), ...]}

    for i, ticker_data in enumerate(universe):
        ticker = ticker_data["ticker"]
        all_pitches[ticker] = {}

        for persona in STOCK_PICKERS:
            result = run_solo_pitch(persona, ticker_data, macro_regime)
            all_pitches[ticker][persona] = result

            if result and result.get("pitch"):
                pitched_tickers.setdefault(ticker, []).append(
                    (persona, result.get("reason", ""))
                )

        # Progress
        n_pitched = len(pitched_tickers)
        if (i + 1) % 25 == 0 or (i + 1) == len(universe):
            print(f"           [{i+1}/{len(universe)}] {n_pitched} tickers pitched so far")

    print(f"           Total pitched: {len(pitched_tickers)} tickers")
    for t, advocates in sorted(pitched_tickers.items()):
        names = [a[0] for a in advocates]
        print(f"             {t}: {', '.join(names)}")

    if not pitched_tickers:
        print("           Nobody pitched anything. Empty watchlist.")
        _save_watchlist(as_of_date, [], macro_regime, all_pitches, {}, {}, t_start)
        return

    # =========================================================================
    # BUILD WATCHLIST (minimum 2 advocates to make the cut)
    # =========================================================================
    MIN_ADVOCATES = 2
    print(f"\n  [STEP 3] Building watchlist (min {MIN_ADVOCATES} advocates)...")

    watchlist = []
    skipped = []
    for ticker, advocates in sorted(pitched_tickers.items()):
        if len(advocates) < MIN_ADVOCATES:
            skipped.append(ticker)
            continue
        ticker_data = next(u for u in universe if u["ticker"] == ticker)
        watchlist.append({
            "ticker": ticker,
            "advocates": [a[0] for a in advocates],
            "reasons": {a[0]: a[1] for a in advocates},
            "close": ticker_data["close"],
            "adx": ticker_data["adx"],
            "rsi": ticker_data["rsi"],
            "rvol": ticker_data["rvol"],
            "zscore_50": ticker_data.get("zscore_50"),
        })
        names = [a[0] for a in advocates]
        print(f"           {ticker:6s} — {', '.join(names)}")

    if skipped:
        print(f"           Dropped {len(skipped)} single-advocate tickers: {', '.join(skipped[:20])}{'...' if len(skipped) > 20 else ''}")

    # Sort by number of advocates (more conviction = higher)
    watchlist.sort(key=lambda x: len(x["advocates"]), reverse=True)

    # =========================================================================
    # SAVE
    # =========================================================================
    _save_watchlist(as_of_date, watchlist, macro_regime, all_pitches, {}, {}, t_start)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"WEEKEND PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"  Watchlist: {len(watchlist)} tickers")
    for w in watchlist:
        print(f"    {w['ticker']:6s} — {', '.join(w['advocates'])}")
    print(f"{'='*70}")


def _save_watchlist(as_of_date, watchlist, macro_regime, pitches, qna, votes, t_start):
    data = {
        "date": as_of_date,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t_start),
        "macro_regime": macro_regime,
        "watchlist": watchlist,
        "pitch_summary": {
            t: {p: r.get("pitch") if r else None for p, r in pv.items()}
            for t, pv in pitches.items()
            if any(r and r.get("pitch") for r in pv.values())
        },
        "qna": qna,
        "votes": votes,
    }
    path = os.path.join(WATCHLIST_DIR, f"watchlist_{as_of_date}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-03-07"
    run_weekend_pipeline(date)
