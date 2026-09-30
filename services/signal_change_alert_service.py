"""
ALR-1: detects a day-over-day change in a ticker's stock_scores
short_signal/long_signal for every user who owns or watchlists it, and
notifies (via services/notification_dispatcher.py) on a genuine change.
Mirrors services/saved_screen_alert_service.py's idiom -- INSERT ... ON
CONFLICT DO NOTHING, notify only on a fresh insert -- rather than
web/backend/portfolio_alerts.py's richer refresh-in-place + LLM-sentiment
shape, since a signal change needs neither.
"""

import logging
from datetime import date

from services.email_service import APP_URL
from services.notification_dispatcher import dispatch_alert
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

UNIVERSE_ID = "All"


async def scan_signal_changes() -> int:
    """Returns the number of alerts dispatched (immediate email or queued
    for digest/quiet-hours, per that user's notification preferences --
    see services/notification_dispatcher.py; the in-app record is
    written regardless, via this function's own INSERT)."""
    today = date.today()

    async with service_conn() as conn:
        # Every (user, ticker) pair from an active portfolio's holdings or
        # a manual watchlist entry -- same owned/watchlisted union shape
        # as web/backend/routers/earnings.py and saved_screen_alert_service.py,
        # done in one cross-user query here instead of per-user.
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

    async with service_conn() as conn:
        latest_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, as_of_date, short_signal, long_signal
            FROM stock_scores WHERE ticker = ANY($1::text[]) AND universe_id = $2
            ORDER BY ticker, as_of_date DESC
            """,
            tickers, UNIVERSE_ID,
        )
    latest_by_ticker = {r["ticker"]: r for r in latest_rows}

    dispatched = 0
    prior_cache: dict[str, dict | None] = {}

    async with service_conn() as conn:
        for row in rows:
            user_id, email, ticker = row["user_id"], row["email"], row["ticker"]
            latest = latest_by_ticker.get(ticker)
            if latest is None or latest["as_of_date"] != today:
                continue  # no fresh capture for this ticker today, nothing to compare

            if ticker not in prior_cache:
                prior_cache[ticker] = await conn.fetchrow(
                    """
                    SELECT short_signal, long_signal FROM stock_scores
                    WHERE ticker = $1 AND universe_id = $2 AND as_of_date < $3
                    ORDER BY as_of_date DESC LIMIT 1
                    """,
                    ticker, UNIVERSE_ID, today,
                )
            prior = prior_cache[ticker]
            if prior is None:
                # First-ever captured day for this ticker -- "changed from
                # nothing" would be a spurious alert, same trap
                # saved_screen_alert_service.py already solved once.
                continue

            changes = []
            if prior["short_signal"] != latest["short_signal"]:
                changes.append(("short", prior["short_signal"], latest["short_signal"]))
            if prior["long_signal"] != latest["long_signal"]:
                changes.append(("long", prior["long_signal"], latest["long_signal"]))

            for horizon, old_signal, new_signal in changes:
                result = await conn.execute(
                    """
                    INSERT INTO signal_change_alerts (user_id, ticker, alert_date, horizon, old_signal, new_signal)
                    VALUES ($1::uuid, $2, $3, $4, $5, $6)
                    ON CONFLICT (user_id, ticker, alert_date, horizon) DO NOTHING
                    """,
                    user_id, ticker, today, horizon, old_signal, new_signal,
                )
                if result == "INSERT 0 1":
                    horizon_label = "Short-term" if horizon == "short" else "Long-term"
                    subject = f"{ticker} {horizon_label.lower()} signal changed: {old_signal} → {new_signal}"
                    text_body = (
                        f"{ticker}'s {horizon_label.lower()} signal changed from {old_signal} to {new_signal} "
                        f"as of today's close.\n\nSee {APP_URL}/stock/{ticker} for the full score breakdown."
                    )
                    values = {"horizon": horizon, "old_signal": old_signal, "new_signal": new_signal}
                    await dispatch_alert(str(user_id), ticker, "signal_change", subject, text_body, values)
                    dispatched += 1

    if dispatched:
        logger.info("Signal change alerts: %d dispatched", dispatched)
    return dispatched
