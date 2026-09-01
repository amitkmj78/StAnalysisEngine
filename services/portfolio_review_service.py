"""
AI-assisted "what needs a look" review across a whole portfolio.

Deliberately built on top of data the app has *already computed* for
each position — the quant model's own Signal (services.signal_
publication_service.compute_predict_algo_comparison, via portfolio
insights), the concentration check, and the Sentiment reading
(services.sentiment_service) — rather than running any fresh analysis.
Flagging itself is plain, deterministic Python (flag_positions below),
so it works even if every LLM provider is down; the LLM's only job is
to turn an already-flagged list into one readable paragraph.

Same "explain, don't instruct" boundary as
services.quant_signal_narrative_service: this synthesizes the app's own
already-computed reads into a summary, it does not generate fresh
buy/sell advice or invent numbers not already shown elsewhere on the
page.
"""

import logging
from typing import Optional

from .llm_setup import invoke_with_fallback

logger = logging.getLogger(__name__)

# Same threshold shown to the user as "concentrated" everywhere else in
# the app (CONCENTRATION_THRESHOLD_PCT in web/backend/routers/
# portfolio.py) — positions already come in pre-flagged via their own
# `concentrated` bool, so this constant isn't re-applied here, just
# documented for context.


def flag_positions(positions: list[dict]) -> list[dict]:
    """
    positions: [{ticker, signal, expected_return_pct, weight_pct,
    concentrated, sentiment_label}, ...] — same shape portfolio insights
    + sentiment already produce, just merged.

    Flags a position (with the specific reason(s) why) when:
    - the quant model's own signal is SELL
    - it's concentrated (the app's existing 25%-of-portfolio check)
    - sentiment reads Bearish while the quant signal isn't BUY either
      (two independent reads agreeing on caution)
    - sentiment reads Bullish and the quant signal is BUY (agreement
      worth surfacing positively too, not just risk)

    Returns only the flagged positions, each with a `reasons` list.
    """
    flagged = []
    for p in positions:
        reasons = []
        if p.get("signal") == "SELL":
            reasons.append("the quant model's signal is SELL")
        if p.get("concentrated"):
            weight = p.get("weight_pct")
            reasons.append(
                f"makes up {weight:.0f}% of the portfolio — concentrated" if weight is not None else "concentrated"
            )
        sentiment = p.get("sentiment_label")
        if sentiment == "Bearish" and p.get("signal") != "BUY":
            reasons.append("sentiment reads Bearish and the signal isn't BUY either")
        if sentiment == "Bullish" and p.get("signal") == "BUY":
            reasons.append("signal and sentiment both read Bullish")
        if reasons:
            flagged.append({**p, "reasons": reasons})
    return flagged


def build_portfolio_review(llms: list, flagged: list[dict]) -> Optional[str]:
    """
    `llms` is an ordered list of available providers, tried in turn via
    invoke_with_fallback. Returns None if `flagged` is empty (caller
    should show a plain "nothing stands out" message, no LLM call
    needed) or if every provider fails.
    """
    if not flagged:
        return None

    lines = []
    for p in flagged:
        expected_return = p.get("expected_return_pct")
        lines.append(
            f"{p['ticker']}: signal={p.get('signal')}, "
            f"expected_return_pct={expected_return if expected_return is not None else 'n/a'}, "
            f"weight_pct={p.get('weight_pct')}, sentiment={p.get('sentiment_label') or 'unavailable'} "
            f"— flagged because: {'; '.join(p['reasons'])}"
        )

    prompt = (
        "Below are positions in a stock portfolio that an automated rules check flagged as worth a "
        "second look, based on this app's own already-computed quant signal, portfolio-concentration "
        "check, and sentiment reading for each (given below — do not invent any numbers not shown "
        "here, and do not consider any other tickers).\n\n"
        + "\n".join(lines)
        + "\n\nWrite a short (3-5 sentence) portfolio review in plain English summarizing what stands "
        "out across these flagged positions and why. Reference specific tickers. Describe what the "
        "data shows and let the reader judge — do not instruct them to buy or sell."
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
        return content
    except Exception as e:
        logger.warning("Portfolio review narrative failed (all providers): %s", e)
        return None
