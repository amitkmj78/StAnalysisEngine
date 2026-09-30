import asyncio
from datetime import date

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from services.stock_detail_service import upcoming_earnings_in_window
from services.yfinance_cache import get_cached_earnings_dates

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/earnings",
    tags=["earnings"],
    dependencies=[Depends(verify_bearer_token)],
)

WINDOW_DAYS = 30


@router.get("/calendar")
@limiter.limit("10/minute")
async def get_earnings_calendar(request: Request):
    """ERN-1: upcoming earnings (next WINDOW_DAYS days) for every ticker
    this user holds or has manually watchlisted. Same owned/watchlisted
    aggregation query pair already used by stock_finder.py's
    _annotate_with_user_state and saved_screen_alert_service.py --
    "watchlisted" deliberately excludes watchlist_alerts rows auto-
    created from an owned position's target/stop (source='portfolio_auto'),
    so it stays a distinct signal from "owned". One yfinance call per
    ticker (no bulk API exists for this), fanned out concurrently and
    backed by get_cached_earnings_dates's own TTL cache -- same
    per-ticker-scan cost profile as Stock Finder's own universe scan."""
    await enforce_daily_quota(request, "earnings/calendar")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        owned_rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM portfolio_positions WHERE user_id = $1::uuid",
            user_id,
        )
        watchlisted_rows = await conn.fetch(
            """
            SELECT DISTINCT ticker FROM watchlist_alerts
            WHERE user_id = $1::uuid AND active AND source IS DISTINCT FROM 'portfolio_auto'
            """,
            user_id,
        )
    owned = {r["ticker"] for r in owned_rows}
    watchlisted = {r["ticker"] for r in watchlisted_rows}
    tickers = sorted(owned | watchlisted)

    if not tickers:
        return {"as_of": date.today().isoformat(), "window_days": WINDOW_DAYS, "entries": []}

    frames = await asyncio.gather(*[run_in_threadpool(get_cached_earnings_dates, t) for t in tickers])

    entries = []
    for ticker, frame in zip(tickers, frames):
        upcoming = upcoming_earnings_in_window(frame, window_days=WINDOW_DAYS)
        if upcoming is None:
            continue
        entries.append(
            {
                "ticker": ticker,
                "owned": ticker in owned,
                "watchlisted": ticker in watchlisted,
                **upcoming,
            }
        )
    entries.sort(key=lambda e: e["date"])

    return {"as_of": date.today().isoformat(), "window_days": WINDOW_DAYS, "entries": entries}
