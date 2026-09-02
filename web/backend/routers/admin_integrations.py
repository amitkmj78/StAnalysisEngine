"""
Admin-only integrations dashboard — lists every external service this
app depends on (LLM providers, market data, search) and lets an admin
run a real live check on demand, rather than only discovering an
outage (e.g. the OpenAI billing exhaustion, the retired Groq model —
both found this session only by SSHing in and grepping logs) when a
real user happens to hit it.

Configuration checks are just "is the env var set" (instant, no I/O).
Tests are real calls — an actual yfinance fetch, an actual LLM
.invoke(), an actual search — so "configured" and "actually working
right now" are reported separately (a configured-but-expired API key
is exactly the case that matters most to catch).
"""

import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from web.backend.admin import require_admin
from web.backend.llm_cache import cached_init_llms, resolve_llm

router = APIRouter(
    prefix="/api/v1/admin/integrations",
    tags=["admin-integrations"],
    dependencies=[Depends(require_admin)],
)


def _test_yfinance() -> str:
    import yfinance as yf

    price = yf.Ticker("AAPL").fast_info.get("lastPrice")
    if price is None:
        raise RuntimeError("No price returned for AAPL — yfinance may be rate-limited or down.")
    return f"Live quote OK — AAPL ${price:.2f}"


def _test_llm(label_prefix: str):
    def run() -> str:
        llm_openai, llm_groq, llm_claude, llm_ollama, labels = cached_init_llms()
        label = next((l for l in labels if l.startswith(label_prefix)), None)
        if label is None:
            raise RuntimeError(f"{label_prefix} is not configured on this server.")
        llm = resolve_llm(label, llm_openai, llm_groq, llm_claude, llm_ollama)
        response = llm.invoke("Reply with exactly one word: OK")
        content = getattr(response, "content", str(response))
        return f"{label} responded: {content.strip()[:120]!r}"

    return run


def _test_brave() -> str:
    from services.web_search.backend import BRAVE_SEARCH_API_KEY, brave_search

    if not BRAVE_SEARCH_API_KEY:
        raise RuntimeError("BRAVE_SEARCH_API_KEY not configured.")
    hits = brave_search("Apple Inc stock", 1)
    if not hits:
        raise RuntimeError("Brave Search returned zero results for a real query — check the API key/quota.")
    return f"Got a result: {hits[0]['title'][:70]!r}"


def _test_duckduckgo() -> str:
    from services.web_search.backend import ddg_search

    hits = ddg_search("Apple Inc stock", 1)
    if not hits:
        raise RuntimeError("DuckDuckGo returned zero results — scraping may be blocked right now.")
    return f"Got a result: {hits[0]['title'][:70]!r}"


def _test_crawlsearch() -> str:
    # Deliberately NOT services.web_search.backend.crawlsearch_search() —
    # that wrapper fails open (returns [] on any exception, by design,
    # so a normal "no results for this query" search doesn't break
    # callers) which would make a real connection failure and a
    # genuine zero-result query look identical here. A direct hit on
    # CrawlSearch's own /health endpoint surfaces a real connection
    # error instead of silently swallowing it.
    import httpx

    from services.web_search.backend import CRAWLSEARCH_API_URL

    if not CRAWLSEARCH_API_URL:
        raise RuntimeError("CRAWLSEARCH_API_URL not configured.")
    response = httpx.get(f"{CRAWLSEARCH_API_URL.rstrip('/')}/health", timeout=10.0)
    response.raise_for_status()
    data = response.json()
    return f"Reachable — {data.get('indexed_pages', '?')} pages indexed"


def _test_alpaca() -> str:
    from services.alpaca_client import get_alpaca_latest_price

    if not (os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")):
        raise RuntimeError("ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY not configured.")
    price = get_alpaca_latest_price("AAPL")
    if price is None:
        raise RuntimeError("No trade returned for AAPL — check the API keys, or IEX may have no recent print.")
    return f"Live quote OK — AAPL ${price:.2f}"


def _test_finnhub() -> str:
    import httpx

    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not configured.")
    response = httpx.get(
        "https://finnhub.io/api/v1/quote", params={"symbol": "AAPL", "token": key}, timeout=10.0
    )
    response.raise_for_status()
    data = response.json()
    price = data.get("c")
    if not price:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return f"Live quote OK — AAPL ${price}"


INTEGRATIONS = {
    "yfinance": {
        "name": "yfinance",
        "category": "Market Data",
        "configured": lambda: True,
        "test": _test_yfinance,
        "note": "No API key — used directly for prices/history throughout the app.",
    },
    "alpaca": {
        "name": "Alpaca",
        "category": "Market Data",
        "configured": lambda: bool(os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")),
        "test": _test_alpaca,
        "note": "Free real-time IEX quote feed — selectable as the live-price provider below.",
    },
    "finnhub": {
        "name": "Finnhub",
        "category": "Market Data",
        "configured": lambda: bool(os.getenv("FINNHUB_API_KEY")),
        "test": _test_finnhub,
        "note": "Optional real-time WebSocket price feed.",
    },
    "groq": {
        "name": "Groq",
        "category": "LLM",
        "configured": lambda: bool(os.getenv("GROQ_API_KEY")),
        "test": _test_llm("Groq"),
        "note": "Default LLM provider (see services/llm_setup.py).",
    },
    "openai": {
        "name": "OpenAI",
        "category": "LLM",
        "configured": lambda: bool(os.getenv("OPENAI_API_KEY")),
        "test": _test_llm("OpenAI"),
        "note": "Second-choice LLM provider / fallback.",
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "category": "LLM",
        "configured": lambda: bool(os.getenv("ANTHROPIC_API_KEY")),
        "test": _test_llm("Claude"),
        "note": "Optional additional LLM provider.",
    },
    "brave": {
        "name": "Brave Search",
        "category": "Search",
        "configured": lambda: bool(os.getenv("BRAVE_SEARCH_API_KEY")),
        "test": _test_brave,
        "note": "Licensed search backend for services/web_search — required for commercial use.",
    },
    "duckduckgo": {
        "name": "DuckDuckGo",
        "category": "Search",
        "configured": lambda: True,
        "test": _test_duckduckgo,
        "note": "No API key — free scraping fallback when Brave/CrawlSearch aren't configured.",
    },
    "crawlsearch": {
        "name": "CrawlSearch",
        "category": "Search",
        "configured": lambda: bool(os.getenv("CRAWLSEARCH_API_URL")),
        "test": _test_crawlsearch,
        "note": "Our own crawler/index — first-choice search source when configured.",
    },
}


def _run_test(entry: dict) -> dict:
    start = time.monotonic()
    try:
        detail = entry["test"]()
        ok = True
    except Exception as e:
        detail = str(e)
        ok = False
    return {"ok": ok, "detail": detail, "latency_ms": int((time.monotonic() - start) * 1000)}


@router.get("")
async def list_integrations():
    return {
        "integrations": [
            {
                "key": key,
                "name": entry["name"],
                "category": entry["category"],
                "configured": entry["configured"](),
                "note": entry["note"],
            }
            for key, entry in INTEGRATIONS.items()
        ]
    }


@router.post("/{key}/test")
async def test_integration(key: str):
    entry = INTEGRATIONS.get(key)
    if entry is None:
        raise HTTPException(404, f"Unknown integration: {key}")
    if not entry["configured"]():
        return {"ok": False, "detail": f"{entry['name']} is not configured on this server.", "latency_ms": None}
    return await run_in_threadpool(_run_test, entry)


# ---------------------------------------------------------------------------
# CrawlSearch on-demand crawl — proxies to the standalone CrawlSearch
# service's own POST/GET/DELETE /api/crawl (see that project's
# crawlsearch/crawl_runner.py). The crawl itself runs entirely on
# CrawlSearch's side (background thread there, Postgres advisory lock
# against its own systemd-timer cycles) — this router just forwards the
# admin's request/poll/stop so it doesn't need to separately open
# CrawlSearch's own UI.
# ---------------------------------------------------------------------------

CRAWLSEARCH_PROXY_TIMEOUT = 10.0  # start/status/stop all return immediately on CrawlSearch's side


def _crawlsearch_base_url() -> str:
    from services.web_search.backend import CRAWLSEARCH_API_URL

    if not CRAWLSEARCH_API_URL:
        raise HTTPException(503, "CRAWLSEARCH_API_URL not configured on this server.")
    return CRAWLSEARCH_API_URL.rstrip("/")


def _crawlsearch_headers() -> dict:
    from services.web_search.backend import CRAWLSEARCH_API_KEY

    return {"X-API-Key": CRAWLSEARCH_API_KEY} if CRAWLSEARCH_API_KEY else {}


def _crawlsearch_request(method: str, path: str, **kwargs) -> dict:
    import httpx

    try:
        response = httpx.request(
            method,
            f"{_crawlsearch_base_url()}{path}",
            headers=_crawlsearch_headers(),
            timeout=CRAWLSEARCH_PROXY_TIMEOUT,
            **kwargs,
        )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach CrawlSearch: {e}") from e
    if response.status_code == 409:
        # CrawlBusy — either an on-demand crawl or the systemd timer's own
        # cycle already holds CrawlSearch's advisory lock.
        detail = response.json().get("detail", "A crawl is already running.")
        raise HTTPException(409, detail)
    response.raise_for_status()
    return response.json()


class CrawlSearchCrawlRequest(BaseModel):
    # None/empty means "every enabled domain" — same contract as
    # CrawlSearch's own POST /api/crawl.
    domains: Optional[list[str]] = None


@router.post("/crawlsearch/crawl")
async def start_crawlsearch_crawl(body: CrawlSearchCrawlRequest):
    """Starts an on-demand CrawlSearch crawl cycle. Runs on CrawlSearch's
    side in the background (minutes to hours for a full cycle) — this
    call itself returns immediately with the initial status snapshot."""
    return await run_in_threadpool(
        _crawlsearch_request, "POST", "/api/crawl", json={"domains": body.domains}
    )


@router.get("/crawlsearch/crawl")
async def get_crawlsearch_crawl_status():
    """Polls the currently running (or most recently finished) crawl's
    live progress — current domain, pages crawled, per-domain results."""
    return await run_in_threadpool(_crawlsearch_request, "GET", "/api/crawl")


@router.delete("/crawlsearch/crawl")
async def stop_crawlsearch_crawl():
    """Requests a cooperative stop. CrawlSearch finishes the fetch already
    in flight and exits between URLs rather than aborting mid-request —
    on purpose, since an aborted fetch mid-response is exactly the kind
    of rudeness its politeness rules exist to prevent — so this returns
    immediately but the run itself ends a moment later; poll GET
    /crawlsearch/crawl to see it land."""
    return await run_in_threadpool(_crawlsearch_request, "DELETE", "/api/crawl")
