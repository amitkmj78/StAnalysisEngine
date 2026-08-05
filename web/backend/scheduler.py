import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from starlette.concurrency import run_in_threadpool

from services.alert_engine_service import evaluate_alert
from services.prediction_verification_service import verify_prediction
from web.backend.app_settings import (
    PIT_PRICE_CAPTURE_ENABLED_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    VERIFY_PREDICTIONS_ENABLED_KEY,
    get_setting_bool,
)
from web.backend.db import service_conn
from web.backend.pit_prices import capture_and_persist_pit_prices
from web.backend.signal_publication import evaluate_due_signal_outcomes, publish_daily_signals

logger = logging.getLogger(__name__)

VERIFY_INTERVAL_MINUTES = 15
ALERT_INTERVAL_MINUTES = 5
# TR-3 Phase 1: capture PIT closes shortly after market close, ahead of
# publication — independent today (nothing consumes this yet), but future
# phases that make publication/comparison PIT-aware will want the day's
# capture already on record before 16:10 ET runs.
PIT_CAPTURE_HOUR_ET = 16
PIT_CAPTURE_MINUTE_ET = 5
# TR-1 / NFR-01: publish within 60 minutes of the US market close (4:00pm ET).
PUBLISH_HOUR_ET = 16
PUBLISH_MINUTE_ET = 10
# TR-4: evaluate outcomes once daily, after publication — no need to check
# more often since "due" is measured in trading days, not minutes.
EVALUATE_HOUR_ET = 17
EVALUATE_MINUTE_ET = 0

_scheduler: AsyncIOScheduler | None = None


async def _verify_all_saved_predictions() -> None:
    """
    Same verify_prediction() logic already used inline by GET /predict/history,
    just run across every user's rows on a schedule instead of only when
    someone happens to revisit the page. Uses service_conn() (bypasses RLS)
    since this isn't scoped to one request's user.
    """
    if not await get_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, default=True):
        logger.info("Scheduler: verify_saved_predictions is disabled, skipping this run")
        return

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


async def _capture_pit_prices_job() -> None:
    """TR-3 Phase 1: append today's close for the tracked universe to the
    point-in-time price store. Gated by its own flag, independent of
    publish_signals_enabled — this is internal data accumulation, not a
    public act, so it's safe to default on."""
    if not await get_setting_bool(PIT_PRICE_CAPTURE_ENABLED_KEY, default=True):
        logger.info("Scheduler: pit_price_capture is disabled, skipping this run")
        return
    inserted = await capture_and_persist_pit_prices()
    if inserted:
        logger.info("Scheduler: captured %d new PIT price rows", inserted)


async def _publish_daily_signals_job() -> None:
    """TR-1: commit today's Signal Set to the public ledger. Off by default
    (see PUBLISH_SIGNALS_ENABLED_KEY) until an admin explicitly enables it —
    deploying this pipeline must not itself be the act that starts the
    public track record. Wraps publish_daily_signals(), which is itself
    idempotent per (target_date, universe, lookback) — safe to run more than
    once (scheduler restart, catching up after a missed run) without
    double-publishing."""
    if not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        logger.info("Scheduler: publish_daily_signals is disabled, skipping this run")
        return
    published = await publish_daily_signals()
    if published:
        logger.info("Scheduler: published %d daily signals", published)


async def _evaluate_signal_outcomes_job() -> None:
    """TR-4: check every published date old enough to have a knowable
    outcome and record it. Gated by the same flag as publication — if
    there's no live record (publishing is off), there's nothing to
    evaluate."""
    if not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        logger.info("Scheduler: publish_daily_signals is disabled, skipping outcome evaluation")
        return
    evaluated = await evaluate_due_signal_outcomes()
    if evaluated:
        logger.info("Scheduler: recorded %d signal outcomes", evaluated)


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
    _scheduler.add_job(
        _capture_pit_prices_job,
        CronTrigger(
            hour=PIT_CAPTURE_HOUR_ET, minute=PIT_CAPTURE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="capture_pit_prices",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _publish_daily_signals_job,
        CronTrigger(
            hour=PUBLISH_HOUR_ET, minute=PUBLISH_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="publish_daily_signals",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,  # catch up if the process was down at 4:10pm ET
    )
    _scheduler.add_job(
        _evaluate_signal_outcomes_job,
        CronTrigger(
            hour=EVALUATE_HOUR_ET, minute=EVALUATE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="evaluate_signal_outcomes",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info(
        "Background scheduler started (verify_saved_predictions every %d min, "
        "evaluate_watchlist_alerts every %d min, capture_pit_prices weekdays %02d:%02d ET, "
        "publish_daily_signals weekdays %02d:%02d ET, evaluate_signal_outcomes weekdays %02d:%02d ET)",
        VERIFY_INTERVAL_MINUTES, ALERT_INTERVAL_MINUTES, PIT_CAPTURE_HOUR_ET, PIT_CAPTURE_MINUTE_ET,
        PUBLISH_HOUR_ET, PUBLISH_MINUTE_ET, EVALUATE_HOUR_ET, EVALUATE_MINUTE_ET,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
