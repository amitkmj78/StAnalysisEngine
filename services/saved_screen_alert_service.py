import json
import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.email_service import send_saved_screen_alert_email
from services.stock_finder_service import apply_filters, rank_stocks
from services.stock_score_capture_service import fetch_latest_scores
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


async def scan_saved_screens_for_membership_changes() -> int:
    """
    SCN-3: for every saved screen, re-applies its own stored goal/universe/
    filters to today's fresh Stock Finder table (same apply_filters logic
    the page.tsx UI itself uses), diffs the resulting ticker set against
    the last recorded membership, and -- only on a genuine enter/leave
    change -- writes a saved_screen_alerts row and emails the screen's
    owner. Mirrors web/backend/portfolio_alerts.py's scan_portfolios_for_
    drops: one row per (screen, day) via ON CONFLICT DO UPDATE, email only
    on the day's row being a fresh INSERT.

    A screen's very first-ever check treats "prior membership" as equal to
    "current membership" (no entered/left, no email) -- otherwise every
    screen would fire a spurious "N stocks just entered" alert the first
    time it's ever scanned, for tickers that matched it all along.

    Returns the number of alert emails actually sent.
    """
    today = date.today()

    async with service_conn() as conn:
        screens = await conn.fetch(
            "SELECT s.*, u.email FROM saved_screens s JOIN users u ON u.id = s.user_id"
        )
    if not screens:
        return 0

    emailed = 0
    rank_cache: dict[tuple[str, str], object] = {}

    async with service_conn() as conn:
        for screen in screens:
            screen_id = screen["id"]
            user_id = screen["user_id"]
            goal, universe = screen["goal"], screen["universe"]
            filters = json.loads(screen["filters"]) if isinstance(screen["filters"], str) else screen["filters"]

            cache_key = (goal, universe)
            if cache_key not in rank_cache:
                rank_cache[cache_key] = await run_in_threadpool(rank_stocks, goal, universe)
            df = rank_cache[cache_key]
            if df.empty:
                continue

            tickers = df["Ticker"].tolist()
            scores = await fetch_latest_scores(tickers)

            owned_rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM portfolio_positions WHERE user_id = $1::uuid", user_id,
            )
            watchlisted_rows = await conn.fetch(
                """
                SELECT DISTINCT ticker FROM watchlist_alerts
                WHERE user_id = $1::uuid AND active AND source IS DISTINCT FROM 'portfolio_auto'
                """,
                user_id,
            )
            owned_tickers = {r["ticker"] for r in owned_rows}
            watchlisted_tickers = {r["ticker"] for r in watchlisted_rows}

            matched = apply_filters(df, scores, filters, owned_tickers, watchlisted_tickers)
            current_set = set(matched)

            prior_row = await conn.fetchrow(
                "SELECT membership FROM saved_screen_alerts WHERE screen_id = $1 ORDER BY check_date DESC LIMIT 1",
                screen_id,
            )
            if prior_row is None:
                prior_set = current_set
            else:
                membership = prior_row["membership"]
                prior_set = set(json.loads(membership) if isinstance(membership, str) else membership)

            entered = sorted(current_set - prior_set)
            left = sorted(prior_set - current_set)

            result = await conn.execute(
                """
                INSERT INTO saved_screen_alerts (user_id, screen_id, check_date, entered, left_tickers, membership)
                VALUES ($1::uuid, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb)
                ON CONFLICT (screen_id, check_date) DO UPDATE SET
                    entered = excluded.entered, left_tickers = excluded.left_tickers, membership = excluded.membership
                """,
                user_id, screen_id, today, json.dumps(entered), json.dumps(left), json.dumps(sorted(current_set)),
            )
            if (entered or left) and result == "INSERT 0 1":
                sent = await run_in_threadpool(
                    send_saved_screen_alert_email, screen["email"], screen["name"], entered, left,
                )
                if sent:
                    emailed += 1

    return emailed
