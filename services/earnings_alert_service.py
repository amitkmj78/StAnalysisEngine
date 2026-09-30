"""
ALR-1: "earnings in 2 days" alert -- one row per (user, ticker, day) this
job runs and finds the ticker inside the 2-day window, emailed once per
day (so a ticker 2 days out and then 1 day out gets two separate daily
emails, each satisfying its own day's "no more than one alert per stock
per type per day" cap -- not deduped to a single "entered the window"
event, since each day's check is its own distinct condition). Reuses
services/stock_detail_service.py::upcoming_earnings_in_window exactly as
ERN-1's own /earnings/calendar does, just with window_days=2 instead of
30 -- no new earnings-window math, only the alert/notify wiring around it.
"""

import asyncio
import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.email_service import APP_URL
from services.notification_dispatcher import dispatch_alert
from services.stock_detail_service import upcoming_earnings_in_window
from services.yfinance_cache import get_cached_earnings_dates
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

EARNINGS_ALERT_WINDOW_DAYS = 2


async def scan_earnings_in_window() -> int:
    """Returns the number of alerts dispatched (see
    services/notification_dispatcher.py)."""
    today = date.today()

    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT u.id AS user_id, u.email, t.ticker
            FROM users u
            JOIN (
                SELECT pp.user_id, pp.ticker FROM portfolio_positions pp
                JOIN portfolios p ON p.id = pp.portfolio_id
                WHERE pp.ticker IS NOT NULL AND p.is_active
                UNION
                SELECT user_id, ticker FROM watchlist_alerts
                WHERE active AND source IS DISTINCT FROM 'portfolio_auto'
            ) t ON t.user_id = u.id
            """
        )
    if not rows:
        return 0

    tickers = sorted({r["ticker"] for r in rows})
    frames = await asyncio.gather(*[run_in_threadpool(get_cached_earnings_dates, t) for t in tickers])

    upcoming_by_ticker: dict[str, dict] = {}
    for ticker, frame in zip(tickers, frames):
        upcoming = upcoming_earnings_in_window(frame, as_of=today, window_days=EARNINGS_ALERT_WINDOW_DAYS)
        if upcoming is not None:
            upcoming_by_ticker[ticker] = upcoming

    if not upcoming_by_ticker:
        return 0

    dispatched = 0
    async with service_conn() as conn:
        for row in rows:
            user_id, ticker = row["user_id"], row["ticker"]
            upcoming = upcoming_by_ticker.get(ticker)
            if upcoming is None:
                continue

            result = await conn.execute(
                """
                INSERT INTO earnings_alert_log (user_id, ticker, alert_date, earnings_date)
                VALUES ($1::uuid, $2, $3, $4)
                ON CONFLICT (user_id, ticker, alert_date) DO NOTHING
                """,
                user_id, ticker, today, date.fromisoformat(upcoming["date"]),
            )
            if result == "INSERT 0 1":
                timing = f" ({upcoming['market_timing']})" if upcoming["market_timing"] else ""
                subject = f"{ticker} reports earnings {upcoming['date']}{timing}"
                text_body = (
                    f"{ticker} is scheduled to report earnings on {upcoming['date']}{timing} -- within "
                    f"{EARNINGS_ALERT_WINDOW_DAYS} days.\n\nSee {APP_URL}/earnings for your full earnings calendar."
                )
                values = {"earnings_date": upcoming["date"], "market_timing": upcoming["market_timing"]}
                await dispatch_alert(str(user_id), ticker, "earnings", subject, text_body, values)
                dispatched += 1

    if dispatched:
        logger.info("Earnings alerts: %d dispatched", dispatched)
    return dispatched
