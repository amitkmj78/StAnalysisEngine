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
import threading
import time
from typing import Optional, TypedDict

import yfinance as yf

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


# Cached 5 minutes on a genuine result — this is one shared, market-
# wide feed, not per-user, so every viewer across every page shares
# one real fetch instead of each triggering their own. A plain
# ttl_cache would cache an EMPTY result (Yahoo rate-limited AND the
# DDG fallback having its own transient off-moment — both observed in
# production) for the same 5 minutes as a real one, leaving the ticker
# dark for the whole window even though a retry seconds later often
# succeeds (confirmed live: DDG returned 0 hits, then 10, within the
# same minute). So an empty/failed fetch is retried almost immediately
# instead of being cached at the success TTL.
SUCCESS_CACHE_TTL_SECONDS = 300
EMPTY_RETRY_SECONDS = 20

_cache_lock = threading.Lock()
_cached_result: Optional[dict] = None
_cached_at: float = 0.0


def get_hot_market_news() -> dict:
    global _cached_result, _cached_at
    now = time.monotonic()
    with _cache_lock:
        if _cached_result is not None:
            ttl = SUCCESS_CACHE_TTL_SECONDS if _cached_result["items"] else EMPTY_RETRY_SECONDS
            if now - _cached_at < ttl:
                return _cached_result

    items = _fetch_yahoo_news()
    source = "yahoo"
    if not items:
        items = _fetch_ddg_fallback()
        source = "duckduckgo"
    result = {"items": items, "source": source}

    with _cache_lock:
        _cached_result = result
        _cached_at = now
    return result
