from fastapi import APIRouter, Depends, Request

from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/filings", tags=["filings"], dependencies=[Depends(verify_bearer_token)])


@router.get("/{ticker}")
@limiter.limit("20/minute")
async def get_filing_summaries(request: Request, ticker: str):
    """SUM-1: the latest stored summary per form type (most recent 10-K,
    most recent 10-Q) for this ticker. Shared per-ticker data, not user-
    scoped -- same precedent as ticker_sentiment_snapshots reads (e.g.
    web/backend/routers/portfolio.py's sentiment endpoint)."""
    await enforce_daily_quota(request, "filings/get")
    ticker = ticker.strip().upper()

    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (form_type) form_type, filing_date, report_date, document_url,
                   compared_to_accession_number, summary, method
            FROM filing_summaries
            WHERE ticker = $1
            ORDER BY form_type, filing_date DESC
            """,
            ticker,
        )

    filings = [
        {
            "form_type": r["form_type"],
            "filing_date": str(r["filing_date"]),
            "report_date": str(r["report_date"]) if r["report_date"] else None,
            "document_url": r["document_url"],
            "compared_to_prior_filing": r["compared_to_accession_number"] is not None,
            "summary": r["summary"],
            "method": r["method"],
        }
        for r in rows
    ]
    return {"ticker": ticker, "filings": filings}
