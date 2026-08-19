import logging
from typing import Any, Dict, Optional

import yfinance as yf

from .llm_setup import invoke_with_fallback
from .prediction_service import generate_trading_signal, predict_future_prices
from .sentiment_service import get_sentiment_summary

logger = logging.getLogger(__name__)

DROP_THRESHOLD_PCT = 1.0
PREDICT_PERIOD = "1y"
PREDICT_DAYS_AHEAD = 10


def get_price_and_prev_close(ticker: str) -> Optional[Dict[str, float]]:
    """
    One fast_info lookup gets both today's live price and yesterday's close
    — the two numbers needed to detect a same-day drop — instead of two
    separate network calls.
    """
    try:
        fast_info = yf.Ticker(ticker).fast_info
        price = fast_info.get("lastPrice")
        prev_close = fast_info.get("previousClose")
        if price is None or prev_close is None or prev_close == 0:
            return None
        return {"price": round(float(price), 4), "prev_close": round(float(prev_close), 4)}
    except Exception as e:
        logger.warning("Portfolio alert: failed to fetch price/prev_close for %s: %s", ticker, e)
        return None


def check_for_drop(ticker: str, threshold_pct: float = DROP_THRESHOLD_PCT) -> Optional[Dict[str, float]]:
    """
    Returns drop details if the ticker is down at least threshold_pct from
    yesterday's close, else None. threshold_pct is positive (e.g. 1.0 means
    "flag a 1% or larger decline").
    """
    quote = get_price_and_prev_close(ticker)
    if quote is None:
        return None
    pct_change = (quote["price"] / quote["prev_close"] - 1.0) * 100
    if pct_change > -threshold_pct:
        return None
    return {"price": quote["price"], "prev_close": quote["prev_close"], "pct_change": round(pct_change, 4)}


def _build_recommendation_prompt(ticker: str, drop: Dict[str, float], sentiment_text: str, signal: Optional[dict]) -> str:
    lines = [
        f"A retail investor's position in {ticker} is down {abs(drop['pct_change']):.2f}% today "
        f"(from {drop['prev_close']} to {drop['price']}). Write a short (3-5 sentence) note on what "
        f"they should consider doing, given the context below. Reference the specific news/earnings "
        f"catalyst if one plausibly explains the move, and weigh it against the quant signal. Frame "
        f"this as considerations for the investor to weigh, not a definitive instruction — do not say "
        f"'you should buy/sell', say what the evidence suggests and let them decide.",
        "",
    ]
    if signal:
        lines.append(
            f"Quant signal ({PREDICT_DAYS_AHEAD}-day horizon): {signal['signal']} "
            f"(expected return {signal['expected_return_pct']}%, target price {signal['target_price']})"
        )
    else:
        lines.append("Quant signal: unavailable for this ticker right now.")
    lines.append("")
    lines.append(f"News & earnings context:\n{sentiment_text}")
    return "\n".join(lines)


def build_drop_analysis(llms: list, ticker: str, drop: Dict[str, float]) -> Dict[str, Any]:
    """
    Given an already-detected drop, gathers sentiment/news and the same
    Predict-page quant signal, then asks the LLM to synthesize a short
    recommended-action note — same "sentiment text -> LLM prompt -> writeup"
    shape as prediction_narrative_service.build_prediction_narrative, just
    anchored on a price-drop event instead of a forecast.

    `llms` is an ordered list of available providers (preferred first)
    — the recommendation call tries each in turn (invoke_with_fallback)
    so one provider being down (rate limit, exhausted billing, an
    outage) doesn't produce "Recommendation unavailable" as long as
    another configured provider is healthy.
    """
    sentiment_text = get_sentiment_summary(ticker, llms=llms)

    signal = None
    try:
        future_df = predict_future_prices(ticker, PREDICT_PERIOD, PREDICT_DAYS_AHEAD, False)
        if future_df is not None and not future_df.empty:
            sig = generate_trading_signal(drop["price"], future_df)
            if "target_price" in sig:
                signal = sig
    except Exception as e:
        logger.warning("Portfolio alert: prediction signal failed for %s: %s", ticker, e)

    prompt = _build_recommendation_prompt(ticker, drop, sentiment_text, signal)
    try:
        recommendation, _ = invoke_with_fallback(llms, prompt)
    except Exception as e:
        logger.warning("Portfolio alert: LLM recommendation failed for %s (all providers): %s", ticker, e)
        recommendation = "Recommendation unavailable — the language model call failed. See news/signal context below."

    return {
        "sentiment_summary": sentiment_text,
        "predicted_signal": signal["signal"] if signal else None,
        "predicted_expected_return_pct": signal["expected_return_pct"] if signal else None,
        "predicted_target_price": signal["target_price"] if signal else None,
        "recommended_action": recommendation,
    }
