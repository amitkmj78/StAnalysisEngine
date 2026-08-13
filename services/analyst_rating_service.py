import logging
from typing import Optional

import yfinance as yf

from .cache_utils import ttl_cache

logger = logging.getLogger(__name__)


def _parse_analyst_info(ticker: str, info: Optional[dict]) -> Optional[dict]:
    """Pure: turns a raw yfinance .info dict into the analyst summary
    shape, or None if this ticker has no analyst coverage. Shared by
    get_analyst_rating_summary (single-ticker, on-demand) and
    pit_analyst_rating_service's bulk capture, so the field mapping can't
    drift between the two call sites."""
    if not info or "recommendationKey" not in info:
        return None

    consensus = info.get("recommendationKey", "n/a").replace("_", " ").title()
    analyst_count = info.get("numberOfAnalystOpinions")

    rec_mean = info.get("recommendationMean")
    # Yahoo's scale runs 1 (Strong Buy) to 5 (Strong Sell) — invert and
    # rescale to a 0-100 "buy%" that reads the intuitive direction.
    buy_pct = round(max(0.0, min(100.0, (5 - rec_mean) / 4 * 100)), 1) if rec_mean else None

    return {
        "ticker": ticker,
        "consensus": consensus,
        "analyst_count": analyst_count,
        "buy_pct": buy_pct,
        "target_mean": info.get("targetMeanPrice"),
        "target_high": info.get("targetHighPrice"),
        "target_low": info.get("targetLowPrice"),
        "current_price": info.get("currentPrice"),
    }


@ttl_cache(maxsize=256, ttl_seconds=3600)
def get_analyst_rating_summary(ticker: str) -> Optional[dict]:
    """
    Real, third-party Wall Street analyst consensus and price targets —
    straight from yfinance, nothing modeled or AI-generated. Returns None
    when yfinance has no analyst coverage for this ticker (common for
    small caps, ETFs, and funds) rather than a fabricated neutral value.
    """
    if not ticker:
        return None

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning("Error fetching analyst info for %s: %s", ticker, e)
        return None

    return _parse_analyst_info(ticker, info)
