"""
Public SDK surface for the self-hosted web search feature. Plain sync
functions — matches every other service in this codebase
(services/sentiment_service.py, services/portfolio_alert_service.py,
etc.), which are all sync and called via run_in_threadpool from async
routers; the HTTP API (web/backend/routers/web_search.py) does exactly
that for this module too.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from .backend import RawSearchHit, discover
from .extract import extract_content

MAX_PARALLEL_FETCHES = 10
CONTENT_SNIPPET_CHARS = 500


@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    response_time_ms: int


def _score(query: str, title: str, content: str) -> float:
    """
    Honest, simple keyword-overlap relevance score (0-1) — a scrape-based
    search backend has no real ranking model behind it, so this doesn't
    pretend to be one. Counts distinct query terms (3+ chars, to skip
    "a"/"is"/etc.) found in the title and content, weighting a title
    match twice as much as a content match.
    """
    terms = {t.lower() for t in query.split() if len(t) > 2}
    if not terms:
        return 0.0
    title_lower = title.lower()
    content_lower = content.lower()
    title_hits = sum(1 for t in terms if t in title_lower)
    content_hits = sum(1 for t in terms if t in content_lower)
    raw = (title_hits * 2 + content_hits) / (len(terms) * 3)
    return round(min(raw, 1.0), 3)


def _build_result(hit: RawSearchHit, query: str, content_already_extracted: bool) -> SearchResult:
    # CrawlSearch hits already carry full article text captured at crawl
    # time (backend.py's discover()) — re-fetching the original URL here
    # would be redundant work and defeat the point of a pre-built index.
    extracted = hit["body"] if content_already_extracted else extract_content(hit["href"])
    body = extracted or hit["body"]
    return SearchResult(
        title=hit["title"],
        url=hit["href"],
        content=body[:CONTENT_SNIPPET_CHARS],
        score=_score(query, hit["title"], body),
        raw_content=extracted,
    )


def search(query: str, max_results: int = 5, include_raw_content: bool = False) -> SearchResponse:
    """
    Finds candidate URLs via backend.discover() — CrawlSearch's own
    curated-domain index when configured, else Brave Search API, else
    DuckDuckGo scraping — then, for sources that don't already provide
    full content, fetches and extracts real article content for each
    result in parallel (extract.py, mirrors
    services/market_data_service.py's _fetch_closes_parallel pattern).
    CrawlSearch hits skip this live-fetch step entirely since their
    content was already extracted at crawl time.

    Content extraction always runs for non-CrawlSearch sources (it's
    what makes `content` a real article snippet instead of a one-line
    blurb); the full extracted text is only *returned* via `raw_content`
    when include_raw_content=True, matching Tavily's own request shape.

    Results are ranked by this module's own score, not DuckDuckGo's
    original order — parallel fetching doesn't preserve that order
    anyway, and re-ranking by extracted-content relevance is the point
    of doing the extraction at all.
    """
    start = time.monotonic()
    hits, content_already_extracted = discover(query, max_results)

    results: list[SearchResult] = []
    if hits:
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_FETCHES, len(hits))) as executor:
            futures = [
                executor.submit(_build_result, hit, query, content_already_extracted) for hit in hits
            ]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: r.score, reverse=True)

    if not include_raw_content:
        for r in results:
            r.raw_content = None

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return SearchResponse(query=query, results=results, response_time_ms=elapsed_ms)


def format_results(response: SearchResponse) -> str:
    """Renders a SearchResponse as readable text for an LLM prompt — the
    same shape callers previously got from stringifying Tavily's result
    list (e.g. services/sentiment_service.py, Agent/newAgent.py)."""
    if not response.results:
        return f"No results found for '{response.query}'."
    return "\n\n".join(f"- {r.title} ({r.url})\n  {r.content}" for r in response.results)


def search_text(query: str, max_results: int = 5) -> str:
    """Convenience wrapper for callers that just want a text blob to drop
    into an LLM prompt, not the structured SearchResponse — this is the
    direct replacement for the old `tavily.run(query)` call pattern used
    throughout this app before Tavily was removed."""
    return format_results(search(query, max_results=max_results, include_raw_content=False))
