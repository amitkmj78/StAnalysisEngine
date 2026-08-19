# web_search

An alternative to [Tavily](https://tavily.com) — search the web and get back clean, extracted article content instead of ten-word snippets. Two discovery backends: **Brave Search API** (licensed, commercial-safe) or **DuckDuckGo scraping** (free, no key, dev/internal use only). Same interface either way — extraction, ranking, and summarization run identically on top of both.

Built for one reason: every "search API for LLMs" product on the market is a thin wrapper around the same idea — query in, extracted content out. The wrapper part doesn't need to be a black box. This package does discovery, real content extraction, transparent ranking, and optional LLM summarization as separate, readable steps.

## Backend: Brave vs DuckDuckGo

`backend.discover(query, max_results)` picks automatically based on config — set `BRAVE_SEARCH_API_KEY` and it's used; unset, it falls back to DuckDuckGo. Nothing downstream (`extract.py`, `client.py`, `summarize.py`) changes based on which one is active.

| | Brave Search API | DuckDuckGo (`ddgs`) |
|---|---|---|
| Cost | ~$3-5 per 1,000 queries | free |
| API key | required (`BRAVE_SEARCH_API_KEY`) | none |
| Terms of service | licensed for commercial/resold use | prohibits scraping/reselling results |
| Reliability | official API, versioned | scraping-based — breaks if DDG's markup changes |
| Use case | **anything you charge money for** | local dev, internal tools, prototyping |

**Why this matters if you're shipping this commercially:** DuckDuckGo's terms don't permit scraping their results for a resold product — fine for internal use, a real legal exposure once you're charging for it. Brave's API is built to be licensed by other products, which is what makes a commercial version of this legitimate. Get a key at [brave.com/search/api](https://brave.com/search/api/).

## What it actually does

1. **Discovery** (`backend.py`) — Brave or DuckDuckGo, candidate result URLs (title, href, one-line snippet). Every search product does this part the same way; it's not the differentiator.
2. **Extraction** (`extract.py`) — fetches each candidate URL and parses out the actual article text (prefers `<article>`/`<main>`, falls back to paragraph-density heuristics; strips nav/ads/scripts). This is the part a raw search backend doesn't give you, and it's the actual value-add over "just call an API yourself."

On top of that:

- **Ranking** (`client.py`) — a transparent keyword-overlap score (title match weighted 2x a content match), not a black-box relevance model. You can read exactly why a result scored what it did.
- **Parallel fetch** — up to 10 concurrent extractions per query (`ThreadPoolExecutor`), so `max_results=5` doesn't mean 5 sequential page loads.
- **Optional LLM summarization** (`summarize.py`) — collapse noisy multi-source results (duplicate headlines, ad copy, truncated snippets) into a 4-8 sentence factual brief before it ever reaches a downstream prompt. Opt-in, since it costs one extra LLM call.
- **Fail-open everywhere** — a broken discovery call, a failed page fetch, or a failed summarization call each degrade to the next-best result rather than raising. A search feature should never be the reason an unrelated request 500s.

## SDK

```python
from services.web_search import search, search_text, search_summary

# Structured result — title, url, extracted content, score, response time
response = search("AAPL earnings guidance", max_results=5)
for r in response.results:
    print(r.score, r.title, r.url)

# Plain text blob, ready to drop into an LLM prompt (direct tavily.run() replacement)
text = search_text("AAPL earnings guidance")

# Search + LLM-cleaned summary in one call (needs an llm with .invoke())
brief = search_summary("AAPL earnings guidance", llm=my_llm)
```

## HTTP API

```
POST /api/v1/websearch/search
Authorization: Bearer <token>
Content-Type: application/json

{"query": "AAPL earnings guidance", "max_results": 5, "include_raw_content": false}
```

Response shape mirrors Tavily's own request/response pattern closely enough to be a genuine swap, not just "similar":

```json
{
  "query": "AAPL earnings guidance",
  "results": [
    {"title": "...", "url": "...", "content": "...", "score": 0.83, "raw_content": null}
  ],
  "response_time_ms": 812
}
```

Auth-gated and quota-limited (`enforce_daily_quota`, 20/minute) — see `web/backend/routers/web_search.py`. There's also a UI at `/web-search` for ad-hoc queries without writing code.

## Configuration

- `BRAVE_SEARCH_API_KEY` — optional. Set to use Brave Search (required for commercial deployment). Unset falls back to DuckDuckGo scraping.

## Dependencies

`ddgs` (DuckDuckGo query client), `httpx` (Brave API calls + page fetch), `beautifulsoup4` (HTML parsing). All already declared in `requirements.txt`.

## Positioning, if you're pulling this out as a standalone commercial product

This package has zero StAnalysisEngine-specific logic — no imports outside `services/rate_limit_utils.py`'s generic backoff helper. It's extractable as-is. With `BRAVE_SEARCH_API_KEY` set, the honest pitch is:

> **A leaner search API for LLM apps.** Same request shape as Tavily, built on Brave's licensed search index — but you also get the full extraction/ranking/summarization pipeline as readable, self-hosted code instead of a black box, and you can run it on your own infra instead of routing every query through another vendor.

Realistic audience: developers who want Tavily-equivalent output with more control and a thinner markup over the underlying search cost, not a claim of better raw search coverage than Brave/Tavily's own index — this package's edge is the extraction and summarization layer, not the discovery layer.

Still not free to run at scale — Brave's per-query cost applies once `BRAVE_SEARCH_API_KEY` is set, this package just doesn't add its own markup on top. The DuckDuckGo fallback stays useful for local dev and internal tooling where that cost and ToS constraint don't apply.
