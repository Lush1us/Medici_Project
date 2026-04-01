"""
Medici — Dummy Agent Runner

Loads a persona's brain.md, feeds it the Step 1 payload, and asks for a trading thesis.
Validates that persona → analysis → structured output works end to end.
"""

import json
import os
import sys
import time
import requests

QWEN_API = "http://localhost:8080/v1/chat/completions"
QWEN_MODEL = "Qwen3.5-4B-Q6_K.gguf"

CONSIGLIO_DIR = os.path.join(os.path.dirname(__file__), "Consiglio")
PAYLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "payloads")

AGENT_SYSTEM = """You are a trading analyst. Your personality, trading style, and preferences are defined below. Stay in character — your biases are part of the analysis.

## Prime Mandate
Your singular focus is maximum portfolio growth. Analyze through your lens, then commit to the highest-conviction bet with aggressive sizing. Concerns exist to sharpen your thesis, not to veto it. Sitting out is not neutral — it costs growth. When your edge is present, bet accordingly and bet big.

## Your Profile
{persona}

## Instructions
You will receive an end-of-day market data payload for a ticker. Analyze it and produce a trading thesis.

Respond in valid JSON only. No markdown. Format:
{{
  "ticker": "SYMBOL",
  "date": "YYYY-MM-DD",
  "bias": "bullish" | "bearish" | "neutral",
  "confidence": 1-10,
  "thesis": "2-4 sentence thesis explaining your read",
  "setup": {{
    "entry_trigger": "what would make you enter",
    "stop_loss": "where you'd place the stop",
    "target": "profit target",
    "timeframe": "expected hold period"
  }},
  "key_indicators": ["which indicators from the payload drove your decision"],
  "concerns": ["what could invalidate this thesis"],
  "would_trade": true | false,
  "why_not": "if would_trade is false, explain why you're sitting out"
}}"""


def load_persona(name: str) -> str:
    path = os.path.join(CONSIGLIO_DIR, name, "brain.md")
    with open(path) as f:
        return f.read()


def load_payload(ticker: str) -> dict:
    # Find the most recent payload for this ticker
    files = [f for f in os.listdir(PAYLOAD_DIR) if f.startswith(ticker + "_")]
    if not files:
        raise FileNotFoundError(f"No payload found for {ticker} in {PAYLOAD_DIR}")
    files.sort(reverse=True)
    path = os.path.join(PAYLOAD_DIR, files[0])
    with open(path) as f:
        return json.load(f)


def run_agent(persona_name: str, ticker: str):
    print(f"\n{'='*60}")
    print(f"AGENT: {persona_name} | TICKER: {ticker}")
    print(f"{'='*60}")

    # Load persona
    print(f"\n[1] Loading persona '{persona_name}'...")
    persona = load_persona(persona_name)
    print(f"    Loaded brain.md ({len(persona)} chars)")

    # Load payload
    print(f"[2] Loading payload for {ticker}...")
    payload = load_payload(ticker)
    print(f"    Date: {payload['date']}")
    print(f"    Regime: {payload['context']['regime']}")
    print(f"    Indicators: {len(payload['indicators'])} available")

    # Build prompt
    system = AGENT_SYSTEM.format(persona=persona)

    # Compact indicators: flatten nested dicts and minify
    compact_ind = {}
    for k, v in payload["indicators"].items():
        if isinstance(v, dict):
            compact_ind[k] = {sk: sv for sk, sv in v.items()}
        else:
            compact_ind[k] = v

    user_msg = f"""End-of-day payload for {ticker} on {payload['date']}:

Market Context:
{json.dumps(payload['context'])}

Available Indicators:
{json.dumps(compact_ind)}

Analyze this data in character and produce your trading thesis."""

    # Call Qwen
    print(f"[3] Running analysis via Qwen...")
    t0 = time.time()

    resp = requests.post(QWEN_API, json={
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,
    }, timeout=300)
    resp.raise_for_status()

    elapsed = time.time() - t0
    raw = resp.json()["choices"][0]["message"]["content"]
    usage = resp.json().get("usage", {})
    print(f"    Done in {elapsed:.1f}s | Tokens: {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out")

    # Parse
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])

    try:
        thesis = json.loads(cleaned.strip())
        valid = True
    except json.JSONDecodeError as e:
        print(f"\n    JSON PARSE FAILED: {e}")
        print(f"    Raw (first 500):\n{raw[:500]}")
        return None

    # Display
    print(f"\n{'='*60}")
    print(f"THESIS — {persona_name}")
    print(f"{'='*60}")
    print(f"  Bias:       {thesis.get('bias', '?')} (confidence {thesis.get('confidence', '?')}/10)")
    print(f"  Would trade: {thesis.get('would_trade', '?')}")
    if not thesis.get("would_trade"):
        print(f"  Why not:    {thesis.get('why_not', '?')}")
    print(f"\n  Thesis: {thesis.get('thesis', '?')}")

    setup = thesis.get("setup", {})
    if thesis.get("would_trade"):
        print(f"\n  Setup:")
        print(f"    Entry:     {setup.get('entry_trigger', '?')}")
        print(f"    Stop:      {setup.get('stop_loss', '?')}")
        print(f"    Target:    {setup.get('target', '?')}")
        print(f"    Timeframe: {setup.get('timeframe', '?')}")

    print(f"\n  Key indicators: {thesis.get('key_indicators', [])}")
    print(f"  Concerns:       {thesis.get('concerns', [])}")

    # Save
    out_dir = os.path.join(CONSIGLIO_DIR, persona_name, "theses")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}_{payload['date']}.json")
    with open(out_path, "w") as f:
        json.dump(thesis, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return thesis


if __name__ == "__main__":
    persona = sys.argv[1] if len(sys.argv) > 1 else "average_joe"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "SPY"
    run_agent(persona, ticker)
