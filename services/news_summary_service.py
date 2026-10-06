"""Signal explanation option 2: an LLM summary of each non-earnings 8-K, written from the filing's own text.

Grounding rule: the summary may only use the filing's text. Every number in the summary must also appear in that text.
If any number does not, the summary is discarded and nothing is stored, so it is retried on a later run. This is the
check that stops an invented figure from reaching a user. Earnings 8-Ks are skipped; they have their own summary.
"""

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup
from starlette.concurrency import run_in_threadpool

from .edgar_service import MAX_SECTION_CHARS, fetch_filing_document_text, get_cik_for_ticker
from .llm_setup import invoke_with_fallback
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")
MIN_TEXT_CHARS = 200


def numbers_in(text: str) -> set[str]:
    """Every number in the text, with thousands separators removed (so "1,234" and "1234" match)."""
    return {m.replace(",", "") for m in NUMBER_PATTERN.findall(text)}


def unsupported_numbers(summary: str, source_text: str) -> set[str]:
    """Numbers the summary states that the filing text does not contain. Empty means the summary is grounded."""
    return numbers_in(summary) - numbers_in(source_text)


def primary_document_from_url(url: str) -> str:
    """The main 8-K document's file name, taken from the filing link (the last path segment)."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def summarize_8k_text(llms: list, ticker: str, filed_on: str, item_labels: str, source_text: str) -> Optional[dict]:
    """{"summary", "method"} when the LLM answers and every number in the answer is in the filing text; else None."""
    truncated_note = " [TRUNCATED]" if len(source_text) >= MAX_SECTION_CHARS else ""
    prompt = (
        f"Below is the text of {ticker}'s SEC 8-K filing dated {filed_on}, which reports: {item_labels}.{truncated_note} "
        "Use only what the text says. Do not invent numbers, names, dates or statements, and do not use outside "
        "knowledge about this company. If the text does not say something, say so plainly.\n\n"
        + source_text
        + "\n\nWrite a short, plain-English summary of what the company announced: who is involved, what changed, "
        "the date it takes effect if given, and any figures stated. Keep it to a few sentences."
    )
    try:
        content, _ = invoke_with_fallback(llms, prompt)
    except Exception as e:
        logger.warning("8-K summary failed for %s (all providers): %s", ticker, e)
        return None

    missing = unsupported_numbers(content, source_text)
    if missing:
        logger.warning("8-K summary for %s dropped: numbers not in the filing text: %s", ticker, sorted(missing)[:5])
        return None
    method = f"LLM summary of the SEC 8-K text dated {filed_on} ({item_labels}).{truncated_note} Every figure checked against the filing."
    return {"summary": content.strip(), "method": method}


async def summarize_pending_8k_news(llms: list, limit: int = 5) -> int:
    """Summarize up to `limit` stored 8-Ks that have no summary yet. Returns how many were stored."""
    async with service_conn() as conn:
        pending = await conn.fetch(
            """
            SELECT n.id, n.url, n.source_key, n.filed_on, n.item_codes, n.event_type,
                   (SELECT min(m.ticker) FROM news_ticker_map m WHERE m.news_id = n.id) AS ticker
            FROM news_items n
            WHERE n.source = 'sec_8k' AND n.event_type <> 'EARNINGS'
              AND NOT EXISTS (SELECT 1 FROM news_item_summaries s WHERE s.news_id = n.id)
            ORDER BY n.filed_on DESC, n.id DESC
            LIMIT $1
            """,
            limit,
        )

    stored = 0
    for item in pending:
        ticker = item["ticker"]
        if ticker is None:
            continue
        cik = await run_in_threadpool(get_cik_for_ticker, ticker)
        if cik is None:
            continue
        html = await run_in_threadpool(
            fetch_filing_document_text, cik, item["source_key"], primary_document_from_url(item["url"])
        )
        if not html:
            continue
        text = BeautifulSoup(html, "html.parser").get_text("\n").strip()[:MAX_SECTION_CHARS]
        if len(text) < MIN_TEXT_CHARS:
            continue
        item_labels = ", ".join(item["item_codes"]) or "no item codes"
        result = await run_in_threadpool(
            summarize_8k_text, llms, ticker, item["filed_on"].isoformat(), item_labels, text
        )
        if result is None:
            continue
        async with service_conn() as conn:
            await conn.execute(
                """
                INSERT INTO news_item_summaries (news_id, summary, method)
                VALUES ($1, $2, $3)
                ON CONFLICT (news_id) DO NOTHING
                """,
                item["id"], result["summary"], result["method"],
            )
        stored += 1
    return stored
