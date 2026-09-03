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
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .data_service import get_effective_price
from .llm_setup import invoke_with_fallback
from .yfinance_cache import get_cached_info

logger = logging.getLogger(__name__)

# Each of these is one live network fetch per ticker (a price quote, an
# .info lookup) — bounded the same way as every other multi-ticker
# fan-out in this codebase (e.g. index_fund_service.MAX_PARALLEL_FETCHES),
# not imported from there since it's an unrelated domain.
MAX_PARALLEL_REVIEW_FETCHES = 4


def compute_market_values(positions: list[dict]) -> dict[str, float]:
    """positions: [{"ticker": str, "shares": float}, ...]. Live prices
    (get_effective_price — after-hours quote when the market's in one of
    those states, else the regular-session price; already ttl_cache'd
    either way, so this is cheap if something else on the page just
    fetched the same ticker), fanned out across a bounded thread pool
    since each fetch is independent I/O. Tickers with no price found are
    omitted, not zeroed — a missing dollar figure should read as
    "unavailable", not $0."""
    if not positions:
        return {}
    tickers = [p["ticker"] for p in positions]
    shares_by_ticker = {p["ticker"]: p.get("shares") or 0 for p in positions}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REVIEW_FETCHES) as executor:
        prices = list(executor.map(get_effective_price, tickers))
    return {
        ticker: shares_by_ticker[ticker] * price
        for ticker, price in zip(tickers, prices)
        if price is not None
    }


def compute_sectors(tickers: list[str]) -> dict[str, str]:
    """Real sector per ticker (get_cached_info — shared 15-minute cache,
    so this doesn't re-fetch a ticker another feature already pulled
    .info for recently), fanned out the same way as compute_market_values.
    Tickers with no sector on file (funds/ETFs, mostly) are omitted."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REVIEW_FETCHES) as executor:
        infos = list(executor.map(get_cached_info, tickers))
    return {
        ticker: info.get("sector")
        for ticker, info in zip(tickers, infos)
        if info and info.get("sector")
    }

# Same threshold shown to the user as "concentrated" everywhere else in
# the app (CONCENTRATION_THRESHOLD_PCT in web/backend/routers/
# portfolio.py) — positions already come in pre-flagged via their own
# `concentrated` bool, so this constant isn't re-applied here, just
# documented for context.

# A single-position check misses concentration spread across several
# tickers in the same sector (e.g. an ETF like XLK plus individual tech
# names) — real exposure to one sector can be much higher than any one
# position's own weight_pct suggests. Set higher than the single-position
# 25% threshold since spreading exposure across multiple names is
# inherently less risky than one ticker at the same weight.
SECTOR_CONCENTRATION_THRESHOLD_PCT = 40.0


def _fmt_dollars(value: Optional[float]) -> str:
    return f"${value:,.0f}" if value is not None else "an unknown amount"


def compute_sector_concentration(positions: list[dict]) -> dict[str, dict]:
    """
    positions: [{ticker, sector, market_value}, ...]. Groups by sector
    (skipping positions with no sector — funds/ETFs and some tickers
    don't carry one) and returns only sectors at or above
    SECTOR_CONCENTRATION_THRESHOLD_PCT of total portfolio market value.

    Returns {sector: {"weight_pct": float, "market_value": float,
    "tickers": [str, ...]}}.
    """
    total = sum(p.get("market_value") or 0.0 for p in positions)
    if total <= 0:
        return {}

    by_sector: dict[str, dict] = {}
    for p in positions:
        sector = p.get("sector")
        market_value = p.get("market_value")
        if not sector or market_value is None:
            continue
        entry = by_sector.setdefault(sector, {"market_value": 0.0, "tickers": []})
        entry["market_value"] += market_value
        entry["tickers"].append(p["ticker"])

    flagged_sectors = {}
    for sector, entry in by_sector.items():
        weight_pct = entry["market_value"] / total * 100.0
        if weight_pct >= SECTOR_CONCENTRATION_THRESHOLD_PCT:
            flagged_sectors[sector] = {
                "weight_pct": round(weight_pct, 2),
                "market_value": round(entry["market_value"], 2),
                "tickers": entry["tickers"],
            }
    return flagged_sectors


def flag_positions(positions: list[dict]) -> list[dict]:
    """
    positions: [{ticker, signal, expected_return_pct, weight_pct,
    concentrated, sentiment_label, market_value, sector}, ...] — same
    shape portfolio insights + sentiment already produce, merged with
    live market_value/sector (market_value and sector are optional —
    older callers/tests can omit them and get the same behavior as
    before sector concentration existed).

    Flags a position (with the specific reason(s) why) when:
    - the quant model's own signal is SELL
    - it's concentrated (the app's existing 25%-of-portfolio check)
    - sentiment reads Bearish while the quant signal isn't BUY either
      (two independent reads agreeing on caution)
    - sentiment reads Bullish and the quant signal is BUY (agreement
      worth surfacing positively too, not just risk)
    - it belongs to a sector that's collectively at or above
      SECTOR_CONCENTRATION_THRESHOLD_PCT of the portfolio, even if this
      one position alone isn't concentrated

    Returns only the flagged positions, each with a `reasons` list.
    """
    sector_flags = compute_sector_concentration(positions)

    flagged = []
    for p in positions:
        reasons = []
        if p.get("signal") == "SELL":
            reasons.append("the quant model's signal is SELL")
        if p.get("concentrated"):
            weight = p.get("weight_pct")
            dollars = _fmt_dollars(p.get("market_value"))
            reasons.append(
                f"makes up {weight:.0f}% ({dollars}) of the portfolio — concentrated"
                if weight is not None
                else "concentrated"
            )
        sentiment = p.get("sentiment_label")
        if sentiment == "Bearish" and p.get("signal") != "BUY":
            reasons.append("sentiment reads Bearish and the signal isn't BUY either")
        if sentiment == "Bullish" and p.get("signal") == "BUY":
            reasons.append("signal and sentiment both read Bullish")

        sector = p.get("sector")
        if sector in sector_flags:
            sf = sector_flags[sector]
            others = [t for t in sf["tickers"] if t != p["ticker"]]
            others_note = f" alongside {', '.join(others)}" if others else ""
            reasons.append(
                f"part of the portfolio's {sector} sector concentration — {sector} makes up "
                f"{sf['weight_pct']:.0f}% ({_fmt_dollars(sf['market_value'])}) of the portfolio "
                f"across {len(sf['tickers'])} position(s){others_note}, even though this position "
                f"alone may not be"
            )

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
        market_value = p.get("market_value")
        lines.append(
            f"{p['ticker']}: signal={p.get('signal')}, "
            f"expected_return_pct={expected_return if expected_return is not None else 'n/a'}, "
            f"weight_pct={p.get('weight_pct')}, position_value={_fmt_dollars(market_value)}, "
            f"sector={p.get('sector') or 'unknown'}, sentiment={p.get('sentiment_label') or 'unavailable'} "
            f"— flagged because: {'; '.join(p['reasons'])}"
        )

    prompt = (
        "Below are positions in a stock portfolio that an automated rules check flagged as worth a "
        "second look, based on this app's own already-computed quant signal, portfolio-concentration "
        "check (both single-position and sector-level), and sentiment reading for each (given below — "
        "do not invent any numbers not shown here, and do not consider any other tickers).\n\n"
        + "\n".join(lines)
        + "\n\nWrite a short (3-5 sentence) portfolio review in plain English summarizing what stands "
        "out across these flagged positions and why, citing the actual dollar amounts and sector "
        "context given above where relevant. Reference specific tickers. Describe what the data shows "
        "and let the reader judge — do not instruct them to buy or sell."
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
        return content
    except Exception as e:
        logger.warning("Portfolio review narrative failed (all providers): %s", e)
        return None
