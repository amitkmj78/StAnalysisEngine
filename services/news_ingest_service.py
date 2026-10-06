"""Signal explanation step 1: SEC 8-K filings stored as news items.

Each 8-K is one news item, keyed by its accession number, so a filing that names several tickers is stored once
and mapped to each of them. The event type comes from the filing's item codes, using the first match in
EVENT_PRIORITY. The title says what the filing is ("Form 8-K: Results of operations ...") and nothing more, because
the item codes are all the filing index gives us without reading the document.

The row-building is pure so it can be tested without a database. The write goes through service_conn.
"""

import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.edgar_service import filing_document_url, get_cik_for_ticker, get_recent_8k_all
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

SOURCE = "sec_8k"
PUBLISHER = "SEC EDGAR"

# Item code -> (event type, plain-English description). Codes not listed here are kept in item_codes but do not set the event.
EVENT_LABELS = {
    "1.01": ("MATERIAL_AGREEMENT", "Entry into a material agreement"),
    "1.02": ("AGREEMENT_TERMINATED", "Termination of a material agreement"),
    "2.01": ("ACQUISITION_OR_DISPOSAL", "Completion of an acquisition or disposal"),
    "2.02": ("EARNINGS", "Results of operations and financial condition"),
    "2.05": ("RESTRUCTURING", "Restructuring or exit costs"),
    "2.06": ("IMPAIRMENT", "Material impairment"),
    "3.01": ("LISTING_NOTICE", "Delisting or listing-standards notice"),
    "4.02": ("RESTATEMENT", "Non-reliance on earlier financial statements"),
    "5.02": ("EXECUTIVE_CHANGE", "Change in directors or officers"),
    "5.07": ("SHAREHOLDER_VOTE", "Shareholder vote results"),
    "7.01": ("REG_FD", "Regulation FD disclosure"),
    "8.01": ("OTHER_EVENT", "Other events"),
    "9.01": ("EXHIBITS", "Financial statements and exhibits"),
}
# Most market-moving first, so a filing with several items is labelled by the most important one.
EVENT_PRIORITY = ["4.02", "2.02", "5.02", "1.01", "2.01", "2.05", "2.06", "1.02", "3.01", "5.07", "7.01", "8.01", "9.01"]


def event_for_items(items: list[str]) -> tuple[str, str]:
    """(event_type, description) for a filing's item codes. Unknown or empty items give OTHER."""
    for code in EVENT_PRIORITY:
        if code in items:
            return EVENT_LABELS[code]
    return ("OTHER", "Form 8-K filing")


def build_news_rows(filings: list[dict], cik: str) -> list[dict]:
    """One news row per filing, ready to insert. Pure: no network or database."""
    rows = []
    for f in filings:
        event_type, description = event_for_items(f["items"])
        rows.append(
            {
                "source": SOURCE,
                "source_key": f["accession_number"],
                "title": f"Form 8-K: {description}",
                "url": filing_document_url(cik, f["accession_number"], f["primary_document"]),
                "publisher": PUBLISHER,
                "filed_on": f["filing_date"],
                "event_type": event_type,
                "item_codes": f["items"],
            }
        )
    return rows


async def ingest_8k_news_for_ticker(ticker: str, days: int = 30) -> int:
    """Fetch the last `days` days of 8-Ks for one ticker, store any new ones, and map them to the ticker.
    Returns how many of this ticker's filings were stored or already present. Filings already stored are
    updated in place, so re-running is safe. An unknown ticker returns 0."""
    cik = await run_in_threadpool(get_cik_for_ticker, ticker)
    if cik is None:
        return 0
    filings = await run_in_threadpool(get_recent_8k_all, cik, days)
    rows = build_news_rows(filings, cik)
    if not rows:
        return 0

    async with service_conn() as conn:
        async with conn.transaction():
            for row in rows:
                # Insert-only: a filing already stored is looked up, never rewritten (the app role has no UPDATE on this table).
                news_id = await conn.fetchval(
                    """
                    INSERT INTO news_items (source, source_key, title, url, publisher, filed_on, event_type, item_codes)
                    VALUES ($1, $2, $3, $4, $5, $6::date, $7, $8)
                    ON CONFLICT (source_key) DO NOTHING
                    RETURNING id
                    """,
                    row["source"], row["source_key"], row["title"], row["url"], row["publisher"],
                    date.fromisoformat(row["filed_on"]), row["event_type"], row["item_codes"],
                )
                if news_id is None:
                    news_id = await conn.fetchval("SELECT id FROM news_items WHERE source_key = $1", row["source_key"])
                await conn.execute(
                    "INSERT INTO news_ticker_map (news_id, ticker) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    news_id, ticker,
                )
    logger.info("news: %s has %d 8-K item(s) in the last %d days", ticker, len(rows), days)
    return len(rows)
