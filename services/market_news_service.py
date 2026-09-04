"""
Broad "hottest news" headline feed for the scrolling ticker at the top
of /portfolio. Deliberately not tied to any one holding — pulled from a
handful of broad-market index/ETF tickers via yfinance's own `.news`
(the same library the rest of the app already gets prices from), so the
headlines reflect "the market" (Fed moves, macro data, broad
selloffs/rallies) rather than one company's earnings.

Falls through to a plain DuckDuckGo search (services.web_search.backend.
ddg_search, already the app's no-API-key search fallback elsewhere) when
yfinance's news comes back empty — a real, observed failure mode for
yfinance endpoints under Yahoo's rate limiting (see get_latest_price's
own fallback, and the get_previous_close caching bug this session).
"""

import logging
from typing import Optional, TypedDict

import yfinance as yf

from .cache_utils import ttl_cache
from .rate_limit_utils import fetch_with_backoff
from .web_search.backend import ddg_search

logger = logging.getLogger(__name__)

# Broad-market bellwethers — index/ETF tickers, not the user's own
# holdings, so the feed stays "market news," not "my portfolio's news."
MARKET_TICKERS = ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"]
MAX_HEADLINES = 20


class NewsItem(TypedDict):
    title: str
    url: str
    source: str
    published_at: Optional[str]


def _fetch_yahoo_news() -> list[NewsItem]:
    items: list[NewsItem] = []
    seen_ids: set[str] = set()
    for ticker in MARKET_TICKERS:
        try:
            raw = fetch_with_backoff(
                lambda t=ticker: yf.Ticker(t).news,
                max_retries=1, base_delay=0.1, retry_delay=1.0,
            )
        except Exception as e:
            logger.warning("Market news: yfinance .news failed for %s: %s", ticker, e)
            continue
        for entry in raw or []:
            content = entry.get("content") or {}
            item_id = entry.get("id") or content.get("id")
            if not item_id or item_id in seen_ids:
                continue
            title = content.get("title")
            # clickThroughUrl (Yahoo's own hosted page) over canonicalUrl
            # (the original publisher) — more consistently reachable
            # across sources than linking straight to dozens of
            # different publisher sites.
            url = (content.get("clickThroughUrl") or {}).get("url") or (content.get("canonicalUrl") or {}).get("url")
            if not title or not url:
                continue
            seen_ids.add(item_id)
            items.append(
                {
                    "title": title,
                    "url": url,
                    "source": (content.get("provider") or {}).get("displayName") or "Yahoo Finance",
                    "published_at": content.get("pubDate"),
                }
            )
    items.sort(key=lambda i: i["published_at"] or "", reverse=True)
    return items[:MAX_HEADLINES]


def _fetch_ddg_fallback() -> list[NewsItem]:
    hits = ddg_search("stock market news today", MAX_HEADLINES)
    return [
        {"title": h["title"], "url": h["href"], "source": "DuckDuckGo", "published_at": None}
        for h in hits
        if h.get("title") and h.get("href")
    ]


@ttl_cache(maxsize=1, ttl_seconds=300)
def get_hot_market_news() -> dict:
    """
    Cached 5 minutes — this is one shared, market-wide feed, not
    per-user, so every viewer of /portfolio shares one real fetch
    instead of each triggering their own.
    """
    items = _fetch_yahoo_news()
    source = "yahoo"
    if not items:
        items = _fetch_ddg_fallback()
        source = "duckduckgo"
    return {"items": items, "source": source}
