"""
SUM-2: summarizes a company's real earnings press release (SEC EDGAR 8-K
Exhibit 99.1) -- NOT the earnings call itself. EDGAR has no transcript or
analyst Q&A; this is deliberately labeled "Earnings Release Summary"
throughout (prompt, stored method text, and the frontend card), never
"Earnings Call Summary", so the gap is disclosed rather than implied away.
See docs/stock-analysis-requirements.html's SUM-2 history for why: a paid
transcript provider (Equibles/Alpha Vantage) would close that gap, but
needs an account/API key only the user can set up -- this ships the free,
account-free EDGAR path now, behind a seam (_PROVIDERS below) a real
provider can slot into later without a rewrite.

Same "do not invent" convention as services/filing_summary_service.py and
services/portfolio_review_service.py -- each service restates its own
guardrail text rather than cross-importing it from Agent/meta_agent.py.
"""

import logging
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup
from starlette.concurrency import run_in_threadpool

from .edgar_service import (
    MAX_SECTION_CHARS,
    fetch_filing_document_text,
    filing_document_url,
    find_exhibit_991_document,
    get_cik_for_ticker,
    get_recent_8k_filings,
)
from .llm_setup import invoke_with_fallback
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

NO_QA_CAVEAT = (
    "Source: SEC EDGAR 8-K Exhibit 99.1 earnings press release only -- this is the company's own "
    "published release, not a transcript of the earnings call, and does not include the analyst "
    "Q&A portion. A real transcript provider isn't configured for this app; the press release is "
    "what's actually available for free."
)


def _fetch_release_text(cik: str, accession_number: str, exhibit_document: str) -> Optional[str]:
    html = fetch_filing_document_text(cik, accession_number, exhibit_document)
    if html is None:
        return None
    return BeautifulSoup(html, "html.parser").get_text("\n").strip()[:MAX_SECTION_CHARS]


# Tried in order; a real paid provider would be inserted before the EDGAR
# fallback, not replace it -- see module docstring. Today there's exactly
# one entry.
_PROVIDERS = [_fetch_release_text]


def fetch_release_source(cik: str, accession_number: str, exhibit_document: str) -> Optional[dict]:
    """{"text": ..., "source_label": ...} from the first provider that
    returns real content, or None if every provider fails/returns nothing."""
    for provider in _PROVIDERS:
        text = provider(cik, accession_number, exhibit_document)
        if text:
            return {"text": text, "source_label": "SEC EDGAR 8-K Exhibit 99.1"}
    return None


def summarize_earnings_release(llms: list, ticker: str, filing_date: str, release_text: str) -> Optional[dict]:
    """Single-document summary (no prior-release diffing -- each release is
    a self-contained announcement, unlike a 10-K/10-Q's "what changed").

    Returns {"summary": str, "method": str} or None if every LLM provider
    fails (caller writes no row -- retried on the next scheduled run)."""
    truncated_note = " [TRUNCATED]" if len(release_text) >= MAX_SECTION_CHARS else ""

    prompt = (
        f"Below is {ticker}'s real earnings press release (SEC EDGAR 8-K Exhibit 99.1), filed "
        f"{filing_date}.{truncated_note} Do not invent any numbers, guidance, or statements not "
        "shown in the text below, and do not use outside/general knowledge about this company. "
        "This text does NOT include analyst Q&A (EDGAR doesn't have it) -- never describe analyst "
        "questions, concerns, or reactions, since none are present in the source. If the release "
        "doesn't address something (e.g. forward guidance), say so plainly rather than guessing.\n\n"
        + release_text
        + "\n\nWrite a short, plain-English summary covering: key reported numbers (revenue, EPS, "
        "etc.), guidance for the next period if given (and whether it's up, down, or unchanged "
        "versus any prior guidance mentioned in this same release), and the overall tone of "
        "management's own commentary in the release."
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
    except Exception as e:
        logger.warning("Earnings release summary failed for %s (all providers): %s", ticker, e)
        return None

    method = f"SEC EDGAR 8-K earnings release for {ticker}, filed {filing_date}.{truncated_note} {NO_QA_CAVEAT}"
    return {"summary": content, "method": method}


async def process_new_earnings_releases_for_ticker(llms: list, ticker: str) -> int:
    """SUM-2 orchestration for one ticker: real EDGAR 8-K (item 2.02) list
    -> skip anything already summarized (by accession_number, idempotent --
    a published release never changes, so this is permanent caching) ->
    find the Exhibit 99.1 press release -> fetch+extract its text ->
    summarize -> store. Returns the number of new rows written, for job
    logging. Uses its own DB connection (web.backend.db.service_conn),
    same precedent as filing_summary_service.process_new_filings_for_ticker."""
    cik = await run_in_threadpool(get_cik_for_ticker, ticker)
    if cik is None:
        return 0

    filings = await run_in_threadpool(get_recent_8k_filings, cik)
    if not filings:
        return 0
    newest = filings[0]

    inserted = 0
    async with service_conn() as conn:
        already = await conn.fetchval(
            "SELECT 1 FROM earnings_release_summaries WHERE ticker = $1 AND accession_number = $2",
            ticker, newest["accession_number"],
        )
        if already:
            return 0

        exhibit = await run_in_threadpool(find_exhibit_991_document, cik, newest["accession_number"])
        if exhibit is None:
            return 0

        source = await run_in_threadpool(fetch_release_source, cik, newest["accession_number"], exhibit)
        if source is None:
            return 0

        result = await run_in_threadpool(summarize_earnings_release, llms, ticker, newest["filing_date"], source["text"])
        if result is None:
            return 0

        document_url = filing_document_url(cik, newest["accession_number"], exhibit)
        await conn.execute(
            """
            INSERT INTO earnings_release_summaries
                (ticker, cik, accession_number, filing_date, report_date, document_url, summary, method)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (ticker, accession_number) DO NOTHING
            """,
            ticker, cik, newest["accession_number"],
            date.fromisoformat(newest["filing_date"]),
            date.fromisoformat(newest["report_date"]) if newest["report_date"] else None,
            document_url, result["summary"], result["method"],
        )
        inserted = 1

    return inserted
