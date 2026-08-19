"""
Raw search-result discovery for services/web_search.

Three sources, same RawSearchHit shape:

- CrawlSearch (crawlsearch_search) — our own standalone crawler+index
  product (../CrawlSearch), covering a curated list of finance/news
  sites. Used first when CRAWLSEARCH_API_URL is set. Its `body` is
  already the full extracted article text captured at crawl time, not a
  snippet — client.py skips its own live extraction for these hits
  (see discover()'s `content_already_extracted` return value), so a hit
  from here costs one internal HTTP call, not a live fetch of the
  original site.
- Brave Search API (brave_search) — licensed, commercially-usable search
  results. Used automatically when BRAVE_SEARCH_API_KEY is set. This is
  what makes the rest of this package (extraction, scoring, summarization
  — services/web_search/README.md) viable to ship as a paid/commercial
  product: Brave's terms permit reselling results as part of a product,
  unlike scraping a search engine's result pages.
- DuckDuckGo via `ddgs` (ddg_search) — free, no API key, scraping-based.
  Kept as the no-key fallback for local dev/internal use, same fragility
  tradeoff as before (breaks if DuckDuckGo changes their page). NOT
  suitable for a commercial deployment on its own — see README.md.

discover() tries them in that order, falling through to the next when a
source is unconfigured or returns nothing (e.g. CrawlSearch's curated
domain list doesn't cover the query's topic) — callers (client.py) use
discover(), not any single source directly, so this priority order
never touches anything downstream.
"""

import logging
import os
from typing import TypedDict

import httpx
from ddgs import DDGS

from services.rate_limit_utils import fetch_with_backoff

logger = logging.getLogger(__name__)

CRAWLSEARCH_API_URL = os.getenv("CRAWLSEARCH_API_URL")
CRAWLSEARCH_API_KEY = os.getenv("CRAWLSEARCH_API_KEY")

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class RawSearchHit(TypedDict):
    title: str
    href: str
    body: str


def crawlsearch_search(query: str, max_results: int) -> list[RawSearchHit]:
    """
    Returns raw hits from our own CrawlSearch index, or an empty list if
    the request fails or nothing in the curated domain list matches
    (caller falls through to the next source either way — never raises).
    """
    try:
        def _fetch() -> httpx.Response:
            response = httpx.post(
                f"{CRAWLSEARCH_API_URL.rstrip('/')}/search",
                json={"query": query, "max_results": max_results},
                headers={"X-API-Key": CRAWLSEARCH_API_KEY} if CRAWLSEARCH_API_KEY else {},
                timeout=10.0,
            )
            response.raise_for_status()
            return response

        response = fetch_with_backoff(_fetch)
        hits = response.json().get("results", [])
    except Exception as e:
        logger.warning("Web search: CrawlSearch query failed for %r: %s", query, e)
        return []
    return [
        {"title": h.get("title") or "", "href": h.get("url") or "", "body": h.get("content") or ""}
        for h in hits
        if h.get("url")
    ]


def brave_search(query: str, max_results: int) -> list[RawSearchHit]:
    """
    Returns raw Brave Search hits for `query`, or an empty list if the
    request fails entirely (caller decides what an empty result set
    means — this never raises). Brave caps `count` at 20 per request.
    """
    try:
        def _fetch() -> httpx.Response:
            response = httpx.get(
                BRAVE_ENDPOINT,
                params={"q": query, "count": min(max_results, 20)},
                headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_SEARCH_API_KEY},
                timeout=10.0,
            )
            response.raise_for_status()
            return response

        response = fetch_with_backoff(_fetch)
        hits = response.json().get("web", {}).get("results", [])
    except Exception as e:
        logger.warning("Web search: Brave query failed for %r: %s", query, e)
        return []
    return [
        {"title": h.get("title") or "", "href": h.get("url") or "", "body": h.get("description") or ""}
        for h in hits
        if h.get("url")
    ]


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


def discover(query: str, max_results: int) -> tuple[list[RawSearchHit], bool]:
    """
    Returns (hits, content_already_extracted). CrawlSearch first (when
    configured), falling through to Brave (when configured), falling
    through to DuckDuckGo. Only CrawlSearch hits carry pre-extracted
    full content, so it's the only source that returns True.
    """
    if CRAWLSEARCH_API_URL:
        hits = crawlsearch_search(query, max_results)
        if hits:
            return hits, True
    if BRAVE_SEARCH_API_KEY:
        return brave_search(query, max_results), False
    return ddg_search(query, max_results), False
