"""A small header ticker: major US/global indices, rates, and crypto,
level + day change, refreshed every 15 minutes (server-cached on that
same cadence, so many open tabs/users share one real yfinance call).
Always Yahoo, regardless of the admin's stock price-source switch --
same reasoning services/market_data_service.py already established
for ^VIX/^VIX3M: these are index/crypto tickers Alpaca has no concept
of at all, confirmed live to silently break once that switch is set
to Alpaca.

^TNX is carried as-is (no /10 conversion) -- services/regime_dimensions.py
already treats this same ticker's value as a plain yield percent
(e.g. 4.25, not 42.5), and this module follows that same precedent.
"""

from __future__ import annotations

import logging

import yfinance as yf

from services.cache_utils import ttl_cache
from services.rate_limit_utils import fetch_with_backoff

logger = logging.getLogger(__name__)

# Labeled for display. The first three are the same major US indices
# services/market_news_service.py's MARKET_TICKERS already treats as
# "the broad market" (minus SPY/QQQ there, which are ETFs tracking two
# of these rather than a fourth distinct index). The rest round out
# the header ticker with small-caps/vol, global markets, rates and
# crypto -- each a market Yahoo covers but Alpaca has no concept of.
MARKET_OVERVIEW_TICKERS = [
    {"ticker": "^GSPC", "label": "S&P 500"},
    {"ticker": "^IXIC", "label": "Nasdaq"},
    {"ticker": "^DJI", "label": "Dow"},
    {"ticker": "^RUT", "label": "Russell 2000"},
    {"ticker": "^VIX", "label": "VIX"},
    {"ticker": "^FTSE", "label": "FTSE 100"},
    {"ticker": "^N225", "label": "Nikkei 225"},
    {"ticker": "^GDAXI", "label": "DAX"},
    {"ticker": "^TNX", "label": "10Y Treasury"},
    {"ticker": "BTC-USD", "label": "Bitcoin"},
    {"ticker": "ETH-USD", "label": "Ethereum"},
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
