import logging
from datetime import date, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from starlette.concurrency import run_in_threadpool

from services.alert_engine_service import evaluate_alert
from services.email_service import APP_URL, send_admin_alert_email, send_rankings_email
from services.prediction_verification_service import verify_prediction
from services.signal_publication_service import DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE
from web.backend.admin import ADMIN_EMAIL
from web.backend.app_settings import (
    DB_BACKUP_ENABLED_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY,
    PIT_PRICE_CAPTURE_ENABLED_KEY,
    PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY,
    PORTFOLIO_DROP_ALERTS_ENABLED_KEY,
    PORTFOLIO_DROP_THRESHOLD_DEFAULT,
    PORTFOLIO_DROP_THRESHOLD_PCT_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    VERIFY_PREDICTIONS_ENABLED_KEY,
    get_setting_bool,
    get_setting_float,
)
from web.backend.db import service_conn
from web.backend.db_backup import run_backup, run_restore_test
from web.backend.pit_prices import (
    capture_and_persist_analyst_ratings,
    capture_and_persist_fundamentals,
    capture_and_persist_pit_prices,
    capture_and_persist_quant_signals,
    capture_and_persist_universe_membership,
)
from web.backend.portfolio_alerts import scan_portfolios_for_drops
from web.backend.signal_publication import (
    evaluate_due_signal_outcomes,
    is_publication_recorded,
    publish_daily_signals,
)

logger = logging.getLogger(__name__)

VERIFY_INTERVAL_MINUTES = 15
ALERT_INTERVAL_MINUTES = 5
# Same-day portfolio drop scan — heavier than watchlist's 5-min interval
# since a fresh drop triggers a sentiment search + LLM call, but still
# needs to catch moves throughout the trading day, not just once at close.
PORTFOLIO_DROP_INTERVAL_MINUTES = 15
# TR-3 Phase 1: capture PIT closes shortly after market close, ahead of
# publication — independent today (nothing consumes this yet), but future
# phases that make publication/comparison PIT-aware will want the day's
# capture already on record before 16:10 ET runs.
PIT_CAPTURE_HOUR_ET = 16
PIT_CAPTURE_MINUTE_ET = 5
# Same time as the price/fundamentals capture above — analyst-rating
# capture is the same cost profile (one network call/ticker), no reason
# to stagger it separately.
PIT_ANALYST_RATING_CAPTURE_HOUR_ET = 16
PIT_ANALYST_RATING_CAPTURE_MINUTE_ET = 7
# Deliberately later in the evening, not alongside the 16:05/16:07 jobs
# above — this one trains a model per ticker (~500 tickers), real CPU
# load, kept separate so it doesn't stack with the network-bound captures
# right at market close when other scheduled/user activity also peaks.
PIT_QUANT_SIGNAL_CAPTURE_HOUR_ET = 18
PIT_QUANT_SIGNAL_CAPTURE_MINUTE_ET = 0
# TR-1 / NFR-01: publish within 60 minutes of the US market close (4:00pm ET).
PUBLISH_HOUR_ET = 16
PUBLISH_MINUTE_ET = 10
# TR-4: evaluate outcomes once daily, after publication — no need to check
# more often since "due" is measured in trading days, not minutes.
EVALUATE_HOUR_ET = 17
EVALUATE_MINUTE_ET = 0
# NFR-01: alert if publication hasn't completed within 60 min of the
# 4:00pm ET close (i.e. by 5:00pm ET).
NFR01_CHECK_HOUR_ET = 17
NFR01_CHECK_MINUTE_ET = 0
# Horizon 1 (RS-3): 10 min after publish, giving _publish_daily_signals_job
# time to actually land today's rows first. Well before the NFR01 60-min
# delayed-publication check at 17:00.
RANKINGS_EMAIL_HOUR_ET = 16
RANKINGS_EMAIL_MINUTE_ET = 20
# NFR-02: escalate if publication is still missing 2 hours after close
# (i.e. by 6:00pm ET) — a genuinely missed day, not just a delay.
NFR02_CHECK_HOUR_ET = 18
NFR02_CHECK_MINUTE_ET = 0
# NFR-03: daily backup at a quiet hour, well after every other job for
# the day has run.
BACKUP_HOUR_ET = 3
BACKUP_MINUTE_ET = 0
# NFR-03: "restore-tested quarterly" as a fully automated job, not a
# human task someone has to remember — runs against the previous night's
# backup, an hour after it completes.
RESTORE_TEST_MONTHS = "1,4,7,10"
RESTORE_TEST_DAY = 1
RESTORE_TEST_HOUR_ET = 4
RESTORE_TEST_MINUTE_ET = 0

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


async def _scan_portfolio_drops_job() -> None:
    """Same-day portfolio drop alerts: for every user's holdings, flag any
    ticker down at least the admin-configured threshold (default 1%) from
    yesterday's close, gather sentiment/news + the Predict-page quant
    signal, and record an in-app recommended-action alert. Off by default
    — unlike the other flags, enabling this starts real per-drop external
    API + LLM spend and user-visible notifications, so it's an admin's
    deliberate opt-in, not a safe-by-default background job."""
    if not await get_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: portfolio_drop_alerts is disabled, skipping this run")
        return
    threshold_pct = await get_setting_float(
        PORTFOLIO_DROP_THRESHOLD_PCT_KEY, default=PORTFOLIO_DROP_THRESHOLD_DEFAULT
    )
    inserted = await scan_portfolios_for_drops(threshold_pct=threshold_pct)
    if inserted:
        logger.info("Scheduler: inserted %d portfolio drop alerts", inserted)


async def _capture_pit_data_job() -> None:
    """TR-3: append today's prices (Phase 1), universe membership (Phase 2),
    and fundamentals (Phase 3) to their respective point-in-time stores. One
    flag gates all three — they're the same "start the clock" concern, just
    different tables — independent of publish_signals_enabled since this is
    internal data accumulation, not a public act, so it's safe to default
    on. Membership runs first (cheapest, no external I/O), then prices,
    then fundamentals (heaviest — one yfinance request per ticker)."""
    if not await get_setting_bool(PIT_PRICE_CAPTURE_ENABLED_KEY, default=True):
        logger.info("Scheduler: pit_price_capture is disabled, skipping this run")
        return

    membership_inserted = await capture_and_persist_universe_membership()
    price_inserted = await capture_and_persist_pit_prices()
    fundamentals_inserted = await capture_and_persist_fundamentals()

    if membership_inserted or price_inserted or fundamentals_inserted:
        logger.info(
            "Scheduler: PIT capture — %d membership rows, %d price rows, %d fundamentals rows",
            membership_inserted, price_inserted, fundamentals_inserted,
        )


async def _capture_pit_analyst_ratings_job() -> None:
    """Appends today's real, third-party analyst consensus (same data as
    the Stock Screener's "Analyst Rating" column) to pit_analyst_rating
    for every ticker in the universe that currently has coverage. Own
    flag, independent of pit_price_capture_enabled — see
    PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY's docstring in app_settings.py."""
    if not await get_setting_bool(PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY, default=True):
        logger.info("Scheduler: pit_analyst_rating_capture is disabled, skipping this run")
        return

    inserted = await capture_and_persist_analyst_ratings()
    if inserted:
        logger.info("Scheduler: PIT analyst rating capture — %d rows", inserted)


async def _capture_pit_quant_signals_job() -> None:
    """Appends today's Quant Signal (same BUY/HOLD/SELL data as /predict
    and the Stock Screener's "Quant Signal" column) to pit_quant_signal
    for every ticker in the universe. CPU-heavy (trains a model per
    ticker) — scheduled later in the evening, separate from the cheaper
    network-bound captures above, and independently pausable via
    PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY."""
    if not await get_setting_bool(PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY, default=True):
        logger.info("Scheduler: pit_quant_signal_capture is disabled, skipping this run")
        return

    inserted = await capture_and_persist_quant_signals()
    if inserted:
        logger.info("Scheduler: PIT quant signal capture — %d rows", inserted)


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


async def _send_rankings_email_job() -> None:
    """Horizon 1 (RS-3): emails today's current rankings to every active
    paid subscriber. Double-gated — off unless BOTH publish_signals and
    horizon1_subscriptions are enabled, since there's no point (and no
    subscribers, in practice) if the feature itself is off. A recipient's
    own email failure is logged and skipped, never allowed to block the
    rest of the batch (same fail-open posture as every other email in
    this app)."""
    if not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        return
    if not await get_setting_bool(HORIZON1_SUBSCRIPTIONS_ENABLED_KEY, default=False):
        return

    async with service_conn() as conn:
        latest = await conn.fetchrow(
            """
            SELECT target_date FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            ORDER BY target_date DESC LIMIT 1
            """,
            DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS,
        )
        if latest is None:
            return
        target_date = latest["target_date"]

        signal_rows = await conn.fetch(
            """
            SELECT rank, ticker, trailing_return_pct FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            target_date, DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS,
        )
        if not signal_rows:
            return
        signals = [dict(r) for r in signal_rows]

        subscribers = await conn.fetch(
            """
            SELECT DISTINCT u.email
            FROM subscriptions s
            JOIN users u ON u.id = s.user_id
            WHERE s.tier = 'paid' AND s.status = 'active'
            """
        )

    sent = 0
    for row in subscribers:
        if await run_in_threadpool(send_rankings_email, row["email"], str(target_date), signals):
            sent += 1
    if subscribers:
        logger.info("Scheduler: rankings email sent to %d/%d active paid subscribers", sent, len(subscribers))


async def check_publication_alert(checkpoint: str, deadline_desc: str, force: bool = False) -> dict:
    """
    Shared check behind both the NFR-01 (60-min) and NFR-02 (2-hour)
    publication deadlines, also reused by the admin manual-test endpoint.
    Only meaningful while publication is actually supposed to be running
    (publish_signals_enabled) — if it's deliberately off, there's nothing
    to alert about, unless force=true bypasses both gates purely to test
    that the email mechanism itself still works. Returns a dict describing
    what happened rather than just a bool, so a manual test call can show
    the admin exactly why an alert did or didn't fire.
    """
    if not force and not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        return {"alert_sent": False, "reason": "publish_signals_enabled is off — nothing to alert about"}

    today = date.today()
    if not force and await is_publication_recorded(today):
        return {"alert_sent": False, "reason": f"publication already recorded for {today.isoformat()}"}

    subject = f"StAnalysisEngine: publication {checkpoint} — {today.isoformat()}"
    body = (
        f"No published signal set has been recorded for {today.isoformat()} as of {deadline_desc} "
        f"after market close.\n\n"
        f"Gate 0 requires ≥ 95% of trading days published, no gaps > 3 days — a silent miss "
        f"here can quietly cost that criterion.\n\n"
        f"Check the scheduler and publish manually if needed: {APP_URL}/admin/scheduler"
    )
    sent = await run_in_threadpool(send_admin_alert_email, ADMIN_EMAIL, subject, body)
    logger.warning(
        "Scheduler: publication %s alert for %s — email %s",
        checkpoint, today.isoformat(), "sent" if sent else "FAILED TO SEND",
    )
    return {"alert_sent": sent, "reason": "publication missing" if sent else "email send failed"}


async def _check_publication_nfr01_job() -> None:
    """NFR-01: alert if today's publication hasn't happened within 60
    minutes of market close (by 5:00pm ET)."""
    await check_publication_alert("delayed (NFR-01, 60-min check)", "60 minutes")


async def _check_publication_nfr02_job() -> None:
    """NFR-02: escalate if today's publication is still missing 2 hours
    after market close (by 6:00pm ET) — a genuinely missed day, not just
    a delay. Deliberately still fires even if the NFR-01 check already
    alerted earlier — a second, distinct-severity notice, not a duplicate."""
    await check_publication_alert("missing (NFR-02, 2-hour check)", "2 hours")


async def _db_backup_job() -> None:
    """NFR-03: dump the published-record + PIT-store tables, upload to S3,
    and run the free structural-integrity check — every night, not just
    quarterly. If it fails, alert the same way a missed publication does,
    since a silently-broken backup is just as much a risk as no backup."""
    if not await get_setting_bool(DB_BACKUP_ENABLED_KEY, default=True):
        logger.info("Scheduler: db_backup is disabled, skipping this run")
        return
    result = await run_backup()
    if result.get("error"):
        logger.warning("Scheduler: NFR-03 backup failed: %s", result["error"])
        await run_in_threadpool(
            send_admin_alert_email, ADMIN_EMAIL,
            "StAnalysisEngine: nightly backup failed",
            f"The nightly NFR-03 backup failed: {result['error']}\n\nCheck {APP_URL}/admin/scheduler",
        )
    else:
        logger.info(
            "Scheduler: NFR-03 backup complete — %s, %d bytes, structural check %s",
            result.get("s3_key"), result.get("size_bytes") or 0,
            "passed" if result.get("structural_check_passed") else "FAILED",
        )


async def _db_restore_test_job() -> None:
    """NFR-03's literal "restore-tested quarterly" requirement, fully
    automated: actually restores the previous night's backup into a
    throwaway database and compares row counts against the live tables,
    rather than relying on a human to remember to do this every quarter."""
    if not await get_setting_bool(DB_BACKUP_ENABLED_KEY, default=True):
        logger.info("Scheduler: db_backup is disabled, skipping quarterly restore test")
        return
    result = await run_restore_test()
    if not result.get("all_match"):
        logger.warning("Scheduler: NFR-03 quarterly restore test failed: %s", result)
        await run_in_threadpool(
            send_admin_alert_email, ADMIN_EMAIL,
            "StAnalysisEngine: quarterly restore test failed",
            f"The quarterly NFR-03 restore test did not pass: {result}\n\nCheck {APP_URL}/admin/scheduler",
        )
    else:
        logger.info("Scheduler: NFR-03 quarterly restore test passed for %s", result.get("s3_key"))


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
        _scan_portfolio_drops_job,
        "interval",
        minutes=PORTFOLIO_DROP_INTERVAL_MINUTES,
        id="scan_portfolio_drops",
        next_run_time=datetime.now(),
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _capture_pit_data_job,
        CronTrigger(
            hour=PIT_CAPTURE_HOUR_ET, minute=PIT_CAPTURE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="capture_pit_data",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _capture_pit_analyst_ratings_job,
        CronTrigger(
            hour=PIT_ANALYST_RATING_CAPTURE_HOUR_ET, minute=PIT_ANALYST_RATING_CAPTURE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="capture_pit_analyst_ratings",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _capture_pit_quant_signals_job,
        CronTrigger(
            hour=PIT_QUANT_SIGNAL_CAPTURE_HOUR_ET, minute=PIT_QUANT_SIGNAL_CAPTURE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="capture_pit_quant_signals",
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
    _scheduler.add_job(
        _send_rankings_email_job,
        CronTrigger(
            hour=RANKINGS_EMAIL_HOUR_ET, minute=RANKINGS_EMAIL_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="send_rankings_email",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _check_publication_nfr01_job,
        CronTrigger(
            hour=NFR01_CHECK_HOUR_ET, minute=NFR01_CHECK_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="check_publication_nfr01",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=1800,
    )
    _scheduler.add_job(
        _check_publication_nfr02_job,
        CronTrigger(
            hour=NFR02_CHECK_HOUR_ET, minute=NFR02_CHECK_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="check_publication_nfr02",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=1800,
    )
    _scheduler.add_job(
        _db_backup_job,
        CronTrigger(
            hour=BACKUP_HOUR_ET, minute=BACKUP_MINUTE_ET,
            timezone="America/New_York",
        ),
        id="db_backup",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _db_restore_test_job,
        CronTrigger(
            month=RESTORE_TEST_MONTHS, day=RESTORE_TEST_DAY,
            hour=RESTORE_TEST_HOUR_ET, minute=RESTORE_TEST_MINUTE_ET,
            timezone="America/New_York",
        ),
        id="db_restore_test",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info(
        "Background scheduler started (verify_saved_predictions every %d min, "
        "evaluate_watchlist_alerts every %d min, scan_portfolio_drops every %d min, "
        "capture_pit_data weekdays %02d:%02d ET, capture_pit_analyst_ratings weekdays %02d:%02d ET, "
        "capture_pit_quant_signals weekdays %02d:%02d ET, publish_daily_signals weekdays %02d:%02d ET, "
        "evaluate_signal_outcomes weekdays %02d:%02d ET, send_rankings_email weekdays %02d:%02d ET, "
        "check_publication_nfr01 weekdays %02d:%02d ET, "
        "check_publication_nfr02 weekdays %02d:%02d ET, db_backup daily %02d:%02d ET, "
        "db_restore_test quarterly (month %s day %d) %02d:%02d ET)",
        VERIFY_INTERVAL_MINUTES, ALERT_INTERVAL_MINUTES, PORTFOLIO_DROP_INTERVAL_MINUTES,
        PIT_CAPTURE_HOUR_ET, PIT_CAPTURE_MINUTE_ET,
        PIT_ANALYST_RATING_CAPTURE_HOUR_ET, PIT_ANALYST_RATING_CAPTURE_MINUTE_ET,
        PIT_QUANT_SIGNAL_CAPTURE_HOUR_ET, PIT_QUANT_SIGNAL_CAPTURE_MINUTE_ET,
        PUBLISH_HOUR_ET, PUBLISH_MINUTE_ET, EVALUATE_HOUR_ET, EVALUATE_MINUTE_ET,
        RANKINGS_EMAIL_HOUR_ET, RANKINGS_EMAIL_MINUTE_ET,
        NFR01_CHECK_HOUR_ET, NFR01_CHECK_MINUTE_ET, NFR02_CHECK_HOUR_ET, NFR02_CHECK_MINUTE_ET,
        BACKUP_HOUR_ET, BACKUP_MINUTE_ET, RESTORE_TEST_MONTHS, RESTORE_TEST_DAY,
        RESTORE_TEST_HOUR_ET, RESTORE_TEST_MINUTE_ET,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
