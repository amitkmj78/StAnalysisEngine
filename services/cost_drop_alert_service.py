"""
ALR-1: "a holding falling a set % from cost" -- distinct from the
existing web/backend/portfolio_alerts.py::scan_portfolios_for_drops,
which compares to *yesterday's close*, not cost basis. Same per-day-cap
idiom (unique(user_id, ticker, alert_date) + ON CONFLICT DO NOTHING +
email-only-on-fresh-insert) but deliberately simpler -- no LLM sentiment
synthesis, no same-day refresh-in-place, matching
services/saved_screen_alert_service.py's lighter shape instead.
"""

import logging
from datetime import date
from typing import Optional

from services.email_service import APP_URL
from services.notification_dispatcher import dispatch_alert
from web.backend.app_settings import COST_DROP_THRESHOLD_DEFAULT, COST_DROP_THRESHOLD_PCT_KEY, get_setting_float
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


async def scan_cost_drops(threshold_pct: Optional[float] = None) -> int:
    """Returns the number of alerts dispatched (see
    services/notification_dispatcher.py). threshold_pct
    mirrors scan_portfolios_for_drops' own contract: None (scheduler's
    normal call) uses the admin-configured global default; an explicit
    float (an admin manual-trigger) overrides it for everyone, for
    testing a hypothetical sensitivity without touching the saved
    setting."""
    today = date.today()
    effective_threshold = (
        threshold_pct
        if threshold_pct is not None
        else await get_setting_float(COST_DROP_THRESHOLD_PCT_KEY, default=COST_DROP_THRESHOLD_DEFAULT)
    )

    async with service_conn() as conn:
        holdings = await conn.fetch(
            """
            SELECT pp.user_id, pp.ticker, pp.avg_cost, pp.current_price, u.email
            FROM portfolio_positions pp
            JOIN users u ON u.id = pp.user_id
            JOIN portfolios p ON p.id = pp.portfolio_id
            WHERE pp.ticker IS NOT NULL AND pp.avg_cost IS NOT NULL AND pp.avg_cost > 0
              AND pp.current_price IS NOT NULL AND p.is_active
            """
        )
        if not holdings:
            return 0
        already_alerted = await conn.fetch(
            "SELECT user_id, ticker FROM cost_drop_alerts WHERE alert_date = $1", today,
        )
    already_alerted_keys = {(r["user_id"], r["ticker"]) for r in already_alerted}

    dispatched = 0
    async with service_conn() as conn:
        for row in holdings:
            key = (row["user_id"], row["ticker"])
            if key in already_alerted_keys:
                continue  # already alerted today -- no refresh-in-place for this simpler alert type

            pct_change = round((row["current_price"] / row["avg_cost"] - 1.0) * 100, 4)
            if pct_change > -effective_threshold:
                continue

            result = await conn.execute(
                """
                INSERT INTO cost_drop_alerts (user_id, ticker, alert_date, avg_cost, current_price, pct_change)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, ticker, alert_date) DO NOTHING
                """,
                row["user_id"], row["ticker"], today, row["avg_cost"], row["current_price"], pct_change,
            )
            if result == "INSERT 0 1":
                subject = f"{row['ticker']} is down {abs(pct_change):.1f}% from your cost basis"
                text_body = (
                    f"{row['ticker']} is now ${row['current_price']:.2f}, down {abs(pct_change):.1f}% from your "
                    f"average cost of ${row['avg_cost']:.2f}.\n\nSee {APP_URL}/portfolio for your full holdings."
                )
                await dispatch_alert(str(row["user_id"]), row["ticker"], "cost_drop", subject, text_body)
                dispatched += 1

    if dispatched:
        logger.info("Cost-drop alerts: %d dispatched", dispatched)
    return dispatched
