"""
SUM-1: turns the real filing section text services.edgar_service extracts
into a "what changed vs the last filing" summary. Same "do not invent"
convention as services.portfolio_review_service's build_portfolio_review/
answer_portfolio_question -- each service restates its own guardrail text
rather than cross-importing it from Agent/meta_agent.py.
"""

import logging
from datetime import date
from typing import Optional

from starlette.concurrency import run_in_threadpool

from .edgar_service import (
    FORM_SECTIONS,
    MAX_SECTION_CHARS,
    extract_item_sections,
    fetch_filing_document_text,
    filing_document_url,
    get_cik_for_ticker,
    get_recent_filings,
)
from .llm_setup import invoke_with_fallback

# Same cross-user/service-role DB-access pattern services/market_regime_
# service.py already uses for its own scheduled daily capture job.
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


def _format_filing(filing: dict) -> str:
    lines = [f"Filed {filing['filing_date']} (report period {filing.get('report_date') or 'unknown'}):"]
    for item, text in filing["sections"].items():
        truncated_note = " [TRUNCATED]" if len(text) >= MAX_SECTION_CHARS else ""
        lines.append(f"\n--- {item}{truncated_note} ---\n{text}")
    return "\n".join(lines)


def summarize_filing(llms: list, ticker: str, current: dict, previous: Optional[dict]) -> Optional[dict]:
    """
    current/previous: {"form", "filing_date", "report_date", "sections":
    {"Item 1A": "...", "Item 7": "...", ...}}. `previous` is None only when
    there's truly no prior filing of this form type on file for this ticker
    (e.g. a recent IPO's first 10-K) -- see services.filing_summary_service's
    caller (web/backend/scheduler.py's job) for how the comparison target is
    resolved; this function never fabricates a "previous" filing.

    Returns {"summary": str, "method": str} or None if every LLM provider
    fails (caller writes no row -- retried on the next scheduled run, same
    as any other fetch failure in this pipeline).
    """
    form = current["form"]
    items_compared = list(current["sections"].keys())
    truncated = [item for item, text in current["sections"].items() if len(text) >= MAX_SECTION_CHARS]
    if previous:
        truncated += [
            item for item, text in previous["sections"].items() if len(text) >= MAX_SECTION_CHARS
        ]

    if previous is None:
        comparison_note = (
            f"No prior {form} filing is on file for {ticker} to compare against -- this is the "
            "earliest one this app has seen for this company."
        )
        filings_block = _format_filing(current)
    else:
        comparison_note = (
            f"Comparing the {form} filed {current['filing_date']} against the prior {form} "
            f"filed {previous['filing_date']}."
        )
        filings_block = (
            f"CURRENT FILING:\n{_format_filing(current)}\n\nPREVIOUS FILING:\n{_format_filing(previous)}"
        )

    prompt = (
        f"Below is real section text extracted from {ticker}'s SEC {form} filing(s) (Items: "
        f"{', '.join(items_compared)}). {comparison_note} Do not invent any numbers, risks, or "
        "guidance not shown in the text below, and do not use outside/general knowledge about this "
        "company. If the current filing doesn't address something (e.g. forward guidance -- SEC "
        "filings are often backward-looking and may not discuss it at all), say so plainly rather "
        "than guessing.\n\n"
        + filings_block
        + "\n\nWrite a short, plain-English summary covering: revenue drivers, margins, new or "
        "materially changed risks, and guidance (if addressed). If there is a previous filing, focus "
        "on what changed rather than restating everything. Reference specific Item numbers for where "
        "each point comes from."
    )

    try:
        content, _ = invoke_with_fallback(llms, prompt)
    except Exception as e:
        logger.warning("Filing summary failed for %s %s (all providers): %s", ticker, form, e)
        return None

    method = (
        f"SEC EDGAR {form} for {ticker}, filed {current['filing_date']}. {comparison_note} "
        f"Sections read: {', '.join(items_compared)}."
    )
    if truncated:
        method += f" Truncated at {MAX_SECTION_CHARS:,} characters for: {', '.join(sorted(set(truncated)))}."
    method += (
        " Links go to the full filing document on SEC.gov -- SEC's filing HTML has no stable "
        "per-section anchor, so this links to the document as a whole rather than the specific "
        "section; the summary above names which Item each point is drawn from instead."
    )

    return {"summary": content, "method": method}


async def process_new_filings_for_ticker(llms: list, ticker: str) -> int:
    """SUM-1 orchestration for one ticker: real EDGAR filing list -> skip
    anything already summarized (by accession_number, idempotent against
    re-runs) -> diff against the prior same-form-type filing straight from
    EDGAR's own filing history (never a previously-stored *summary* --
    always a fresh fetch of the real prior filing's text, so summaries
    never compound on top of each other) -> summarize -> store. Returns the
    number of new rows written, for job logging. Uses its own DB connection
    (web.backend.db.service_conn) rather than taking one from the caller --
    same precedent as services/market_regime_service.py's scheduler-job
    entry point."""
    cik = await run_in_threadpool(get_cik_for_ticker, ticker)
    if cik is None:
        return 0

    filings = await run_in_threadpool(get_recent_filings, cik, tuple(FORM_SECTIONS.keys()))

    inserted = 0
    async with service_conn() as conn:
        for form_type, items in FORM_SECTIONS.items():
            same_form = [f for f in filings if f["form"] == form_type]
            if not same_form:
                continue
            newest = same_form[0]

            already = await conn.fetchval(
                "SELECT 1 FROM filing_summaries WHERE ticker = $1 AND accession_number = $2",
                ticker, newest["accession_number"],
            )
            if already:
                continue

            current_html = await run_in_threadpool(
                fetch_filing_document_text, cik, newest["accession_number"], newest["primary_document"]
            )
            if current_html is None:
                continue
            current = {
                "form": form_type,
                "filing_date": newest["filing_date"],
                "report_date": newest["report_date"],
                "sections": await run_in_threadpool(extract_item_sections, current_html, items),
            }

            previous = None
            prior_accession = None
            if len(same_form) > 1:
                prior = same_form[1]
                prior_html = await run_in_threadpool(
                    fetch_filing_document_text, cik, prior["accession_number"], prior["primary_document"]
                )
                if prior_html is not None:
                    previous = {
                        "form": form_type,
                        "filing_date": prior["filing_date"],
                        "report_date": prior["report_date"],
                        "sections": await run_in_threadpool(extract_item_sections, prior_html, items),
                    }
                    prior_accession = prior["accession_number"]

            result = await run_in_threadpool(summarize_filing, llms, ticker, current, previous)
            if result is None:
                continue

            document_url = filing_document_url(cik, newest["accession_number"], newest["primary_document"])
            await conn.execute(
                """
                INSERT INTO filing_summaries
                    (ticker, cik, form_type, accession_number, filing_date, report_date,
                     document_url, compared_to_accession_number, summary, method)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (ticker, accession_number) DO NOTHING
                """,
                ticker, cik, form_type, newest["accession_number"],
                date.fromisoformat(newest["filing_date"]),
                date.fromisoformat(newest["report_date"]) if newest["report_date"] else None,
                document_url, prior_accession, result["summary"], result["method"],
            )
            inserted += 1

    return inserted
