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
) -> Optional[str]:
    """
    `llms` is an ordered list of available providers (preferred first),
    tried in turn via invoke_with_fallback so one provider being down
    doesn't take down this narrative as long as another is healthy.
    Returns None on total failure (every provider failed), matching the
    existing contract callers already handle (see web/backend/routers/
    signals.py's `if narrative is None: raise HTTPException(502, ...)`).
    """
    indicators_text = _current_indicators_text(ticker)

    prompt = (
        f"The internal quant model's current signal for {ticker} is {signal}, "
        f"with an expected {expected_return_pct:+.2f}% return over the next 10 trading days "
        f"(target price {target_price}, last close {last_close}).\n\n"
        f"Current technical picture: {indicators_text}\n\n"
        "In 2-4 sentences, explain what in this technical picture is plausibly consistent with "
        "that signal — momentum, trend direction, overbought/oversold positioning, or volatility "
        "band position. Base this only on the technical indicators given, not on analyst opinions, "
        "news, or earnings. Describe what the indicators show and let the reader judge — do not "
        "instruct them to buy or sell."
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
        return content
    except Exception as e:
        logger.warning("Quant signal narrative failed for %s (all providers): %s", ticker, e)
        return None
