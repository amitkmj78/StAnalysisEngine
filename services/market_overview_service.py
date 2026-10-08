"""A small header ticker: S&P 500 / Nasdaq / Dow level + day change,
refreshed every 15 minutes (server-cached on that same cadence, so
many open tabs/users share one real yfinance call). Always Yahoo,
regardless of the admin's stock price-source switch -- same reasoning
services/market_data_service.py already established for ^VIX/^VIX3M:
these are index tickers Alpaca has no concept of at all, confirmed
live to silently break once that switch is set to Alpaca.
"""

from __future__ import annotations

import logging

import yfinance as yf

from services.cache_utils import ttl_cache
from services.rate_limit_utils import fetch_with_backoff

logger = logging.getLogger(__name__)

# Labeled for display -- same three major US indices
# services/market_news_service.py's MARKET_TICKERS already treats as
# "the broad market" (minus SPY/QQQ there, which are ETFs tracking two
# of these rather than a fourth distinct index).
MARKET_OVERVIEW_TICKERS = [
    {"ticker": "^GSPC", "label": "S&P 500"},
    {"ticker": "^IXIC", "label": "Nasdaq"},
    {"ticker": "^DJI", "label": "Dow"},
]

CACHE_TTL_SECONDS = 15 * 60


@ttl_cache(maxsize=4, ttl_seconds=CACHE_TTL_SECONDS)
def get_market_overview() -> list[dict]:
    out = []
    for entry in MARKET_OVERVIEW_TICKERS:
        ticker = entry["ticker"]
        price = change_pct = None
        try:
            info = fetch_with_backoff(
                lambda t=ticker: yf.Ticker(t).fast_info, max_retries=1, base_delay=0.1, retry_delay=1.0
            )
            price = info.get("lastPrice")
            prev_close = info.get("previousClose")
            if price is not None and prev_close:
                change_pct = round((price / prev_close - 1) * 100, 2)
        except Exception as e:
            logger.warning("Market overview: failed to fetch %s: %s", ticker, e)
        out.append({"ticker": ticker, "label": entry["label"], "price": price, "change_pct": change_pct})
    return out
