"""
Raw search-result discovery for services/web_search — DuckDuckGo via the
`ddgs` package (free, no API key, no new paid vendor). This is
intentionally the "dumb" half of the system: it only finds candidate
URLs (title/href/body snippet), same as any search engine's results
page. The genuinely self-built part — real content extraction and
scoring — lives in extract.py/client.py, not here.

Unofficial/scraping-based, same category of fragility this app already
accepts elsewhere (services/stock_finder_service.py's Wikipedia S&P 500
list) — if DuckDuckGo changes their page, this can start failing until
`ddgs` is updated upstream. fetch_with_backoff gives it the same
rate-limit resilience already used for yfinance.
"""

import logging
from typing import TypedDict

from ddgs import DDGS

from services.rate_limit_utils import fetch_with_backoff

logger = logging.getLogger(__name__)


class RawSearchHit(TypedDict):
    title: str
    href: str
    body: str


def ddg_search(query: str, max_results: int) -> list[RawSearchHit]:
    """
    Returns raw DuckDuckGo hits for `query`, or an empty list if the
    search fails entirely (caller decides what an empty result set
    means — this never raises).
    """
    try:
        hits = fetch_with_backoff(lambda: DDGS().text(query, max_results=max_results))
    except Exception as e:
        logger.warning("Web search: DuckDuckGo query failed for %r: %s", query, e)
        return []
    return [
        {"title": h.get("title") or "", "href": h.get("href") or "", "body": h.get("body") or ""}
        for h in hits
        if h.get("href")
    ]
