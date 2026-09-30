"""
ALR-2: the single funnel every ALR alert-producing service calls instead
of emailing directly. Resolves the user's preference for this
alert_type (a ticker-specific row overrides the global "ticker IS NULL"
row; no row at all falls back to a sane default so alerts work before a
user ever visits a settings page), then either sends immediately or
queues into pending_digest_items when the user is in quiet hours or has
digest mode on -- one hourly job (web/backend/scheduler.py::
_flush_pending_digest_job) flushes both cases later with one
consolidated email.

In-app delivery needs no branch here at all: every alert table
(watchlist_alerts, portfolio_drop_alerts, signal_change_alerts,
earnings_alert_log, cost_drop_alerts) already IS the in-app record --
the caller's own INSERT happened before dispatch_alert was ever called,
so "disabled" below only ever means "don't email/queue", never "don't
record".

ALR-3: also fires an outbound webhook (services/webhook_service.py) when
the user has one configured, independent of email preferences/quiet-
hours/digest -- a power user's own automation is a separate delivery
channel from the ones those settings govern.
"""

from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

from starlette.concurrency import run_in_threadpool

from services.email_service import APP_URL, send_alert_email
from services.webhook_service import send_webhook
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")

# Public: also used by web/backend/routers/alert_preferences.py to show
# users what "no override saved" actually resolves to.
DEFAULT_PREFERENCE = {"enabled": True, "channel_email": True, "channel_inapp": True}


async def _resolve_preference(user_id: str, ticker: Optional[str], alert_type: str) -> dict:
    async with service_conn() as conn:
        row = None
        if ticker is not None:
            row = await conn.fetchrow(
                """
                SELECT enabled, channel_email, channel_inapp FROM user_alert_preferences
                WHERE user_id = $1::uuid AND ticker = $2 AND alert_type = $3
                """,
                user_id, ticker, alert_type,
            )
        if row is None:
            row = await conn.fetchrow(
                """
                SELECT enabled, channel_email, channel_inapp FROM user_alert_preferences
                WHERE user_id = $1::uuid AND ticker IS NULL AND alert_type = $2
                """,
                user_id, alert_type,
            )
    if row is None:
        return dict(DEFAULT_PREFERENCE)
    return {"enabled": row["enabled"], "channel_email": row["channel_email"], "channel_inapp": row["channel_inapp"]}


def is_within_quiet_hours(now_et: datetime, start: Optional[time], end: Optional[time]) -> bool:
    """Handles the overnight-wraparound case (e.g. 22:00-07:00) as well
    as a same-day window (e.g. 09:00-17:00) -- wraparound is whenever
    start > end. No quiet hours configured (either bound unset) means
    never in quiet hours."""
    if start is None or end is None:
        return False
    current = now_et.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


async def dispatch_alert(
    user_id: str,
    ticker: Optional[str],
    alert_type: str,
    subject: str,
    text_body: str,
    values: Optional[dict] = None,
) -> None:
    """Called by every ALR alert-producing service in place of emailing
    directly, right after that service's own fresh-insert check (the
    unique(user_id, ticker, alert_date, ...) constraint on each alert
    table is what enforces ALR-2's "no more than one alert per stock per
    type per day" -- this function doesn't need its own cap logic since
    it's only ever called once per genuine new event).

    `values` is ALR-3's structured payload data (e.g. {"pct_change": ...,
    "avg_cost": ...}) -- optional since not every caller has passed it
    yet, but required for a webhook to fire (an email-only alert with no
    values simply never reaches send_webhook below)."""
    async with service_conn() as conn:
        settings_row = await conn.fetchrow(
            """
            SELECT quiet_hours_start, quiet_hours_end, digest_enabled,
                   webhook_enabled, webhook_url, webhook_secret
            FROM user_notification_settings WHERE user_id = $1::uuid
            """,
            user_id,
        )

    # ALR-3: webhooks fire independently of quiet hours/digest/email
    # preference -- a power user piping alerts into their own automation
    # wants real-time delivery, and their downstream system can filter
    # or rate-limit itself; the acceptance criteria for quiet hours/
    # digest only ever mention push/email/in-app.
    if settings_row and settings_row["webhook_enabled"] and settings_row["webhook_url"] and settings_row["webhook_secret"]:
        payload = {
            "ticker": ticker,
            "alert_type": alert_type,
            "values": values or {},
            "link": f"{APP_URL}/stock/{ticker}" if ticker else None,
        }
        sent = await run_in_threadpool(
            send_webhook, settings_row["webhook_url"], settings_row["webhook_secret"], payload
        )
        if not sent:
            logger.warning("dispatch_alert: webhook delivery failed for user %s (%s)", user_id, alert_type)

    preference = await _resolve_preference(user_id, ticker, alert_type)
    if not preference["enabled"] or not preference["channel_email"]:
        return

    now_et = datetime.now(EASTERN)
    in_digest_mode = bool(settings_row and settings_row["digest_enabled"])
    in_quiet_hours = bool(settings_row) and is_within_quiet_hours(
        now_et, settings_row["quiet_hours_start"], settings_row["quiet_hours_end"]
    )

    if in_digest_mode or in_quiet_hours:
        async with service_conn() as conn:
            await conn.execute(
                """
                INSERT INTO pending_digest_items (user_id, ticker, alert_type, subject, text_body)
                VALUES ($1::uuid, $2, $3, $4, $5)
                """,
                user_id, ticker, alert_type, subject, text_body,
            )
        return

    async with service_conn() as conn:
        to_email = await conn.fetchval("SELECT email FROM users WHERE id = $1::uuid", user_id)
    if to_email:
        sent = await run_in_threadpool(send_alert_email, to_email, subject, text_body)
        if not sent:
            logger.warning("dispatch_alert: email send failed for %s (%s)", to_email, alert_type)
