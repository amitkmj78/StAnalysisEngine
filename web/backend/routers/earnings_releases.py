from fastapi import APIRouter, Depends, Request

from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/earnings-releases", tags=["earnings-releases"], dependencies=[Depends(verify_bearer_token)]
)


@router.get("/{ticker}")
@limiter.limit("20/minute")
async def get_earnings_release_summary(request: Request, ticker: str):
    """SUM-2: the single latest stored earnings-release summary for this
    ticker. Shared per-ticker data, not user-scoped -- same precedent as
    web/backend/routers/filings.py's SUM-1 read endpoint."""
    await enforce_daily_quota(request, "earnings-releases/get")
    ticker = ticker.strip().upper()

    async with service_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT filing_date, report_date, document_url, summary, method
            FROM earnings_release_summaries
            WHERE ticker = $1
            ORDER BY filing_date DESC
            LIMIT 1
            """,
            ticker,
        )

    if row is None:
        return {"ticker": ticker, "release": None}

    return {
        "ticker": ticker,
        "release": {
            "filing_date": str(row["filing_date"]),
            "report_date": str(row["report_date"]) if row["report_date"] else None,
            "document_url": row["document_url"],
            "summary": row["summary"],
            "method": row["method"],
        },
    }
