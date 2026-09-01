"""
On-demand AI explanation for an already-known Quant Signal (BUY/HOLD/
SELL from pit_quant_signal), for the new quant-vs-analyst comparison
page. Deliberately narrow — explains the technical picture only, never
analyst opinions or news sentiment, per the explicit "for Quant Signal
only" scope. Does NOT re-run predict_future_prices/generate_trading_signal
(that would risk producing a narrative for a *different* signal than the
one already shown, since the PIT value was captured earlier that day) —
takes the signal as input and explains it against current technicals.
"""

import logging
from typing import Optional

from .data_service import get_stock_data
from .llm_setup import invoke_with_fallback
from .technical_service import add_indicators

logger = logging.getLogger(__name__)


def _current_indicators_text(ticker: str) -> str:
    data = get_stock_data(ticker, "6mo")
    if data.empty:
        return "Technical indicator data unavailable for this ticker right now."
    df = add_indicators(data)
    if df.empty:
        return "Not enough price history to compute technical indicators yet."
    last = df.iloc[-1]
    return (
        f"RSI(14): {last['RSI']:.1f}. "
        f"MACD: {last['MACD']:.2f} vs signal line {last['MACD_signal']:.2f}. "
        f"20-day SMA: {last['SMA20']:.2f}, 50-day SMA: {last['SMA50']:.2f}. "
        f"Bollinger band range: {last['BB_low']:.2f} - {last['BB_high']:.2f}, "
        f"last close {last['Close']:.2f}."
    )


def build_quant_signal_narrative(
    llms: list,
    ticker: str,
    signal: str,
    expected_return_pct: float,
    target_price: float,
    last_close: float,
    current_price: Optional[float] = None,
) -> Optional[dict]:
    """
    `llms` is an ordered list of available providers (preferred first),
    tried in turn via invoke_with_fallback so one provider being down
    doesn't take down this narrative as long as another is healthy.
    Returns None on total failure (every provider failed), matching the
    existing contract callers already handle (see web/backend/routers/
    signals.py's `if narrative is None: raise HTTPException(502, ...)`).

    Returns {"technical": str, "plain_english": str} — the technical
    version for a reader who already knows RSI/MACD/Bollinger, the plain
    version translating the same facts into everyday language for one who
    doesn't. Both explain, neither instructs ("do not tell the reader to
    buy or sell") — that boundary is deliberate, not an oversight: this
    function only interprets an already-computed signal (BUY/HOLD/SELL is
    decided elsewhere, by the model, not by this LLM call), and a plain-
    language rewrite must preserve that same restraint rather than
    smuggling in fresh advice just because the wording got simpler.

    `current_price` (optional, a live quote the caller fetched separately
    — this function never fetches one itself) lets the prompt address
    whether the move since the signal was captured (last_close) is
    tracking the model's call or has diverged from it, using the same
    current technical picture — still descriptive, not a claim about
    what caused the move, since correlation with today's indicators
    isn't causation.
    """
    indicators_text = _current_indicators_text(ticker)

    outcome_context = ""
    if current_price is not None and last_close:
        actual_pct_change = (current_price - last_close) / last_close * 100.0
        outcome_context = (
            f"\n\nSince this signal was captured, the price has actually moved from {last_close} to "
            f"{current_price} ({actual_pct_change:+.2f}%), against a model forecast of "
            f"{expected_return_pct:+.2f}% over 10 trading days. Note whether the current technical "
            f"picture above still resembles what it looked like when the signal was likely generated, "
            f"or has shifted — but do not claim to know why the price moved the way it did; the "
            f"indicators only describe where things stand now, not a cause for what already happened."
        )

    prompt = (
        f"The internal quant model's current signal for {ticker} is {signal}, "
        f"with an expected {expected_return_pct:+.2f}% return over the next 10 trading days "
        f"(target price {target_price}, last close {last_close})."
        f"{outcome_context}\n\n"
        f"Current technical picture: {indicators_text}\n\n"
        "Write two short explanations of what in this technical picture is plausibly consistent "
        "with that signal — momentum, trend direction, overbought/oversold positioning, or "
        "volatility band position." + (
            " If an actual price move since the signal is given above, also note in each version "
            "whether the current picture still supports the original call or looks different now."
            if outcome_context
            else ""
        ) + " Base both only on the technical indicators (and the actual price move, if given) — not on "
        "analyst opinions, news, or earnings. Describe what the indicators show and let the "
        "reader judge — do not instruct them to buy or sell, and do not add any new opinion "
        "beyond what the indicators show.\n\n"
        "Respond in exactly this two-part format, nothing else:\n"
        "TECHNICAL: <2-4 sentences, for a reader who already knows RSI, MACD, moving averages, "
        "and Bollinger Bands>\n"
        "PLAIN ENGLISH: <2-4 sentences, same facts, no jargon at all — explain what each signal "
        "means in everyday terms (e.g. \"oversold\" as \"sold off harder than usual, possibly due "
        "for a bounce\") as if to someone who has never looked at a stock chart>"
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
    except Exception as e:
        logger.warning("Quant signal narrative failed for %s (all providers): %s", ticker, e)
        return None

    technical = None
    plain_english = None
    current_key = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TECHNICAL:"):
            current_key = "technical"
            technical = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("PLAIN ENGLISH:"):
            current_key = "plain_english"
            plain_english = stripped.split(":", 1)[1].strip()
        elif stripped and current_key == "technical":
            technical = f"{technical} {stripped}".strip()
        elif stripped and current_key == "plain_english":
            plain_english = f"{plain_english} {stripped}".strip()

    if technical is None and plain_english is None:
        # Model didn't follow the format — fall back to showing the whole
        # response as the technical field rather than discarding a real
        # answer just because it wasn't split into two labeled parts.
        return {"technical": content.strip(), "plain_english": None}
    return {"technical": technical, "plain_english": plain_english}
