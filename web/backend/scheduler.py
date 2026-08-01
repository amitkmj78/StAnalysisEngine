import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from starlette.concurrency import run_in_threadpool

from services.alert_engine_service import evaluate_alert
from services.prediction_verification_service import verify_prediction
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

VERIFY_INTERVAL_MINUTES = 15
ALERT_INTERVAL_MINUTES = 5

_scheduler: AsyncIOScheduler | None = None


async def _verify_all_saved_predictions() -> None:
    """
    Same verify_prediction() logic already used inline by GET /predict/history,
    just run across every user's rows on a schedule instead of only when
    someone happens to revisit the page. Uses service_conn() (bypasses RLS)
    since this isn't scoped to one request's user.
    """
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM saved_predictions WHERE verified_at IS NULL")
        if not rows:
            return

        checked = 0
        updated = 0
        for row in rows:
            row_dict = dict(row)
            updates = await run_in_threadpool(verify_prediction, row_dict)
            checked += 1
            if not updates:
                continue
            set_cols = list(updates.keys())
            set_clause = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(set_cols))
            values = [updates[col] for col in set_cols]
            await conn.execute(
                f"UPDATE saved_predictions SET {set_clause} WHERE id = $1",
                row["id"], *values,
            )
            updated += 1

    logger.info("Scheduler: checked %d unverified saved predictions, updated %d", checked, updated)


async def _evaluate_watchlist_alerts() -> None:
    """Second scheduler job: check every not-yet-triggered watchlist alert's
    condition against a live price, across every user, on its own (shorter)
    interval since price moves faster than what the prediction-verify job
    cares about."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM watchlist_alerts WHERE triggered_at IS NULL")
        if not rows:
            return

        checked = 0
        triggered = 0
        for row in rows:
            price = await run_in_threadpool(
                evaluate_alert, row["ticker"], row["condition_type"], row["threshold"]
            )
            checked += 1
            if price is None:
                continue
            await conn.execute(
                "UPDATE watchlist_alerts SET triggered_at = now(), triggered_price = $2 WHERE id = $1",
                row["id"], price,
            )
            triggered += 1

    logger.info("Scheduler: checked %d watchlist alerts, triggered %d", checked, triggered)


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _verify_all_saved_predictions,
        "interval",
        minutes=VERIFY_INTERVAL_MINUTES,
        id="verify_saved_predictions",
        next_run_time=datetime.now(),  # also run once immediately on startup
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _evaluate_watchlist_alerts,
        "interval",
        minutes=ALERT_INTERVAL_MINUTES,
        id="evaluate_watchlist_alerts",
        next_run_time=datetime.now(),
        coalesce=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(
        "Background scheduler started (verify_saved_predictions every %d min, evaluate_watchlist_alerts every %d min)",
        VERIFY_INTERVAL_MINUTES, ALERT_INTERVAL_MINUTES,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
