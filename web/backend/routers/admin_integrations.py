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

from fastapi import APIRouter, Depends, HTTPException
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
