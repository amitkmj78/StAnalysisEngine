"""Signal explanation step 1: the stored news for one ticker. Public, like the other per-stock read endpoints; the rows
are public SEC filings, so there is nothing user-specific here."""

from fastapi import APIRouter, Query, Request

from web.backend.db import service_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/news", tags=["news"])


@router.get("/{ticker}")
@limiter.limit("60/minute")
async def get_news_for_ticker(request: Request, ticker: str, days: int = Query(7, ge=1, le=90)):
    ticker = ticker.upper()
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT n.id, n.title, n.url, n.publisher, n.filed_on, n.event_type, n.item_codes,
                   COALESCE(e.summary, s.summary) AS summary_text,
                   COALESCE(e.method, s.method) AS summary_method
            FROM news_items n
            JOIN news_ticker_map m ON m.news_id = n.id
            -- An earnings 8-K (Item 2.02) links to its press-release summary, when one has been written.
            LEFT JOIN earnings_release_summaries e ON e.ticker = m.ticker AND e.accession_number = n.source_key
            -- Any other 8-K can carry an LLM summary written from its own text.
            LEFT JOIN news_item_summaries s ON s.news_id = n.id
            WHERE m.ticker = $1 AND n.filed_on >= CURRENT_DATE - $2::int
            ORDER BY n.filed_on DESC, n.id DESC
            """,
            ticker, days,
        )
    return {
        "ticker": ticker,
        "days": days,
        "items": [
            {
                "id": r["id"],
                "title": r["title"],
                "url": r["url"],
                "publisher": r["publisher"],
                # A filing carries a date only, so no time of day is shown.
                "filed_on": r["filed_on"].isoformat(),
                "event_type": r["event_type"],
                "item_codes": r["item_codes"],
                # Present only when a summary has been stored for this filing.
                "summary": r["summary_text"],
                "summary_method": r["summary_method"],
            }
            for r in rows
        ],
    }
