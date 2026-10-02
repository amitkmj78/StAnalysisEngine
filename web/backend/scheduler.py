import logging
from datetime import date, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from starlette.concurrency import run_in_threadpool

from services.alert_engine_service import evaluate_alert
from services.basket_rebalance_service import scan_baskets_for_rebalance
from services.cost_drop_alert_service import scan_cost_drops
from services.earnings_alert_service import scan_earnings_in_window
from services.email_service import APP_URL, send_admin_alert_email, send_digest_email, send_rankings_email
from services.filing_summary_service import process_new_filings_for_ticker
from services.market_regime_service import compute_and_persist_daily_regime
from services.notification_dispatcher import EASTERN, is_within_quiet_hours
from services.prediction_verification_service import verify_prediction
from services.signal_change_alert_service import scan_signal_changes
from services.saved_screen_alert_service import scan_saved_screens_for_membership_changes
from services.signal_publication_service import DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE, TRACK_RECORD_HORIZONS
from services.stock_finder_service import SP500_UNIVERSE_NAME, get_stock_finder_table
from services.stock_score_capture_service import compute_and_persist_daily_scores
from web.backend.admin import ADMIN_EMAIL
from web.backend.app_settings import (
    BASKET_REBALANCE_ENABLED_KEY,
    COST_DROP_ALERTS_ENABLED_KEY,
    DB_BACKUP_ENABLED_KEY,
    EARNINGS_ALERTS_ENABLED_KEY,
    FILING_SUMMARIES_ENABLED_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    MARKET_REGIME_ENABLED_KEY,
    PAPER_TRADING_ENABLED_KEY,
    PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY,
    PIT_PRICE_CAPTURE_ENABLED_KEY,
    PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY,
    PORTFOLIO_DROP_ALERTS_ENABLED_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    SAVED_SCREEN_ALERTS_ENABLED_KEY,
    SIGNAL_CHANGE_ALERTS_ENABLED_KEY,
    STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY,
    STOCK_SCORE_COMPUTE_ENABLED_KEY,
    VERIFY_PREDICTIONS_ENABLED_KEY,
    get_setting_bool,
)
from web.backend.db import service_conn
from web.backend.db_backup import run_backup, run_restore_test
from web.backend.llm_cache import cached_init_llms, ordered_llms
from web.backend.paper_order_sync import poll_open_orders, sync_positions_and_cash
from web.backend.pit_prices import (
    capture_and_persist_analyst_ratings,
    capture_and_persist_fundamentals,
    capture_and_persist_pit_prices,
    capture_and_persist_quant_signals,
    capture_and_persist_universe_membership,
    evaluate_due_quant_signal_outcomes,
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
# NFR-1's 2s status-propagation target isn't met by polling -- this is the
# accepted v1 gap (see services/alpaca_trading_client.py's module docstring
# and the paper-trading plan doc). 10s keeps it close without hammering
# Alpaca's rate limits across every linked paper account.
PAPER_ORDER_POLL_INTERVAL_SECONDS = 10
PAPER_POSITIONS_SYNC_INTERVAL_MINUTES = 5
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
# Phase 1 ("Trust") two-score system: runs after that day's prices/
# fundamentals/universe membership are on record (16:05 ET above), well
# before the unrelated 18:00 ET quant-signal capture -- these scores are
# a rules-based composite of already-captured PIT data, not the GBM
# quant signal, so they don't need to wait for it.
STOCK_SCORE_COMPUTE_HOUR_ET = 16
STOCK_SCORE_COMPUTE_MINUTE_ET = 15
# Deliberately later in the evening, not alongside the 16:05/16:07 jobs
# above — this one trains a model per ticker (~500 tickers), real CPU
# load, kept separate so it doesn't stack with the network-bound captures
# right at market close when other scheduled/user activity also peaks.
PIT_QUANT_SIGNAL_CAPTURE_HOUR_ET = 18
PIT_QUANT_SIGNAL_CAPTURE_MINUTE_ET = 0
# REG-1/2/3: 10 minutes after the quant-signal capture above — this job
# fans out ~500+8 tickers' full price history (services/
# market_data_service.py::fetch_market_internals_history), throttled to
# MAX_PARALLEL_FETCHES=4 after a real prior rate-limit incident, so it's
# deliberately kept off the network-bound 16:0x captures and spaced past
# the CPU-heavy quant-signal job rather than stacked with either.
MARKET_REGIME_HOUR_ET = 18
MARKET_REGIME_MINUTE_ET = 10
# SUM-1: after market close, so same-day filings are caught by the next
# run (filings mostly post during the trading day or shortly after).
FILING_SUMMARIES_HOUR_ET = 20
FILING_SUMMARIES_MINUTE_ET = 0
# After the capture above lands today's rows — mostly picks up older
# calls that just became due (a call is only evaluable once horizon_days
# *trading* days have actually elapsed, so this rarely evaluates today's
# own capture; see evaluate_due_quant_signal_outcomes).
QUANT_SIGNAL_EVALUATE_HOUR_ET = 18
QUANT_SIGNAL_EVALUATE_MINUTE_ET = 30
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
# Diversified Basket rebalance checks: 'monthly'-frequency baskets are
# checked every time this fires (the 1st of every month); 'quarterly'
# ones only in the same 4 months as the DB restore-test above — reusing
# this file's own month-gating idiom (CronTrigger's own month= field)
# rather than per-portfolio due-date arithmetic against arbitrary
# creation dates.
BASKET_REBALANCE_HOUR_ET = 7
BASKET_REBALANCE_MINUTE_ET = 0
BASKET_REBALANCE_QUARTERLY_MONTHS = "1,4,7,10"
# Keeps get_stock_finder_table's 1-hour TTL cache warm for the two
# universes the Diversified Basket page's <3s generation goal actually
# needs fast ("All", S&P 500) -- see services/stock_finder_service.py's
# module docstring for the underlying cold-scan timing. Interval must
# stay under that 3600s TTL or the cache still goes cold between runs.
STOCK_FINDER_PREWARM_INTERVAL_MINUTES = 50
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
    condition against a live price or today's short_score, across every
    user, on its own (shorter) interval since price moves faster than
    what the prediction-verify job cares about."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM watchlist_alerts WHERE triggered_at IS NULL")
        if not rows:
            return

        # ALR-1: stock_scores only changes once/day, so batch-fetch every
        # score-condition ticker's latest short_score in one query rather
        # than per-row -- same "one bulk query, not N" idiom as
        # web/backend/routers/portfolio.py::_attach_stock_forecasts.
        score_tickers = sorted({r["ticker"] for r in rows if r["condition_type"] in ("score_above", "score_below")})
        scores_by_ticker: dict[str, float] = {}
        if score_tickers:
            score_rows = await conn.fetch(
                """
                SELECT DISTINCT ON (ticker) ticker, short_score
                FROM stock_scores WHERE ticker = ANY($1::text[]) AND universe_id = 'All'
                ORDER BY ticker, as_of_date DESC
                """,
                score_tickers,
            )
            scores_by_ticker = {r["ticker"]: r["short_score"] for r in score_rows if r["short_score"] is not None}

        checked = 0
        triggered = 0
        for row in rows:
            matched_value = await run_in_threadpool(
                evaluate_alert, row["ticker"], row["condition_type"], row["threshold"],
                scores_by_ticker.get(row["ticker"]),
            )
            checked += 1
            if matched_value is None:
                continue
            # triggered_price holds whatever value satisfied the
            # condition -- a live price for price_* alerts, today's
            # short_score for score_* alerts.
            await conn.execute(
                "UPDATE watchlist_alerts SET triggered_at = now(), triggered_price = $2 WHERE id = $1",
                row["id"], matched_value,
            )
            triggered += 1

    logger.info("Scheduler: checked %d watchlist alerts, triggered %d", checked, triggered)


async def _scan_portfolio_drops_job() -> None:
    """Same-day portfolio drop alerts: for every user's holdings, flags any
    ticker down at least their configured threshold (or the admin-
    configured default) from yesterday's close, gathers sentiment/news +
    the Predict-page quant signal, and records an in-app recommended-
    action alert plus an email to that holding's owner. Off by default —
    unlike the other flags, enabling this starts real per-drop external
    API + LLM spend, so it's an admin's deliberate opt-in, not a
    safe-by-default background job.

    Still scans every user, not just the admin: the in-app UI/API for
    viewing and self-configuring drop alerts is admin-only now (see
    web/backend/routers/portfolio.py), but the admin can still turn
    monitoring on or off for any specific user's portfolio from the
    admin Users screen (POST /admin/users/{id}/portfolios/{id}/
    drop-alerts) — that only has an effect if the scan still looks at
    every user's (admin-controlled) drop_alerts_enabled flag rather than
    only the admin's own."""
    if not await get_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: portfolio_drop_alerts is disabled, skipping this run")
        return
    inserted = await scan_portfolios_for_drops()
    if inserted:
        logger.info("Scheduler: inserted %d portfolio drop alerts", inserted)


async def _scan_cost_drop_alerts_job() -> None:
    """ALR-1: "a holding falling a set % from cost" -- distinct from
    _scan_portfolio_drops_job above, which compares to yesterday's close.
    Same interval rationale as that job (needs to catch moves throughout
    the trading day), deliberately simpler (no LLM synthesis). Off by
    default, same opt-in posture as portfolio_drop_alerts."""
    if not await get_setting_bool(COST_DROP_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: cost_drop_alerts is disabled, skipping this run")
        return
    emailed = await scan_cost_drops()
    if emailed:
        logger.info("Scheduler: %d cost-drop alert emails sent", emailed)


async def _flush_pending_digest_job() -> None:
    """ALR-2: hourly check -- for each user with queued
    pending_digest_items (see services/notification_dispatcher.py::
    dispatch_alert), sends one consolidated email when either (a)
    they're in digest mode and the current ET hour matches their
    digest_time's hour, or (b) they're not in digest mode (their items
    were queued only because dispatch_alert caught them in quiet hours)
    and the current time is now outside their quiet-hours window. No
    gate flag -- this only ever sends what dispatch_alert already
    decided to queue, same "always on" posture as
    _evaluate_watchlist_alerts above."""
    now_et = datetime.now(EASTERN)

    async with service_conn() as conn:
        user_rows = await conn.fetch("SELECT DISTINCT user_id FROM pending_digest_items WHERE flushed_at IS NULL")
    if not user_rows:
        return

    flushed_users = 0
    for row in user_rows:
        user_id = row["user_id"]
        async with service_conn() as conn:
            settings_row = await conn.fetchrow(
                """
                SELECT digest_enabled, digest_time, quiet_hours_start, quiet_hours_end
                FROM user_notification_settings WHERE user_id = $1::uuid
                """,
                user_id,
            )
            email = await conn.fetchval("SELECT email FROM users WHERE id = $1::uuid", user_id)

        if settings_row and settings_row["digest_enabled"]:
            should_flush = settings_row["digest_time"] is not None and now_et.hour == settings_row["digest_time"].hour
        elif settings_row:
            should_flush = not is_within_quiet_hours(
                now_et, settings_row["quiet_hours_start"], settings_row["quiet_hours_end"]
            )
        else:
            # No settings row at all shouldn't happen (dispatch_alert only
            # queues when one exists and triggered digest/quiet-hours),
            # but flush rather than strand items indefinitely if it does.
            should_flush = True

        if not should_flush or not email:
            continue

        async with service_conn() as conn:
            items = await conn.fetch(
                """
                SELECT id, ticker, alert_type, subject, text_body FROM pending_digest_items
                WHERE user_id = $1::uuid AND flushed_at IS NULL ORDER BY created_at
                """,
                user_id,
            )
            if not items:
                continue
            sent = await run_in_threadpool(send_digest_email, email, [dict(i) for i in items])
            if sent:
                ids = [i["id"] for i in items]
                await conn.execute("UPDATE pending_digest_items SET flushed_at = now() WHERE id = ANY($1::bigint[])", ids)
                flushed_users += 1

    if flushed_users:
        logger.info("Scheduler: digest flushed for %d users", flushed_users)


async def _poll_paper_orders_job() -> None:
    """Reconciles every paper-trading account with an order still in a
    SUBMITTING/OPEN/PARTIALLY_FILLED local status against Alpaca's own
    view, and re-syncs that account's positions/cash whenever anything
    changed. Off by default -- an admin must explicitly enable paper
    trading before this starts making Alpaca API calls."""
    if not await get_setting_bool(PAPER_TRADING_ENABLED_KEY, default=False):
        return
    async with service_conn() as conn:
        has_open_orders = await conn.fetchval(
            "SELECT exists(SELECT 1 FROM paper_orders WHERE status IN ('SUBMITTING','OPEN','PARTIALLY_FILLED'))"
        )
    if not has_open_orders:
        return
    changed = await poll_open_orders()
    if changed:
        logger.info("Scheduler: reconciled %d paper order status changes", changed)


async def _sync_paper_positions_job() -> None:
    """Periodic positions/cash sync for every linked paper-trading account
    (TRD-26), independent of the order-status poller above so a fill that
    happens between polls still gets picked up on a fixed cadence."""
    if not await get_setting_bool(PAPER_TRADING_ENABLED_KEY, default=False):
        return
    synced = await sync_positions_and_cash()
    if synced:
        logger.info("Scheduler: synced %d paper-trading positions", synced)


async def _check_basket_rebalances_monthly_job() -> None:
    """Re-ranks each 'monthly'-frequency Diversified Basket's original
    universe/goal, flags weight drift past its own threshold and any
    holding that fell out of its sector's fresh top-N, and writes a
    review-and-act alert -- never executes a trade itself. Off by
    default, same admin-opt-in rationale as portfolio drop alerts."""
    if not await get_setting_bool(BASKET_REBALANCE_ENABLED_KEY, default=False):
        logger.info("Scheduler: basket_rebalance_enabled is disabled, skipping this run")
        return
    inserted = await scan_baskets_for_rebalance(rebalance_frequency_filter="monthly")
    if inserted:
        logger.info("Scheduler: inserted/refreshed %d monthly basket rebalance alerts", inserted)


async def _check_basket_rebalances_quarterly_job() -> None:
    """Same as the monthly job above, but for 'quarterly'-frequency
    baskets — a separate job/trigger (month-restricted via CronTrigger)
    rather than one job doing its own month arithmetic in Python."""
    if not await get_setting_bool(BASKET_REBALANCE_ENABLED_KEY, default=False):
        logger.info("Scheduler: basket_rebalance_enabled is disabled, skipping this run")
        return
    inserted = await scan_baskets_for_rebalance(rebalance_frequency_filter="quarterly")
    if inserted:
        logger.info("Scheduler: inserted/refreshed %d quarterly basket rebalance alerts", inserted)


async def _prewarm_stock_finder_cache_job() -> None:
    """Off by default: a real, continuous increase in steady-state Yahoo
    Finance traffic (a full universe scan roughly every 50 minutes,
    forever) purely to keep the Diversified Basket page's generation
    fast in steady state -- a cost/rate-limit-exposure tradeoff an admin
    should opt into deliberately."""
    if not await get_setting_bool(STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY, default=False):
        return
    await run_in_threadpool(get_stock_finder_table, "All")
    await run_in_threadpool(get_stock_finder_table, SP500_UNIVERSE_NAME)
    logger.info("Scheduler: stock-finder cache prewarmed for 'All' and %r", SP500_UNIVERSE_NAME)


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


async def _compute_stock_scores_job() -> None:
    """Phase 1 ("Trust"): computes and persists today's short-term/
    long-term composite scores for every ticker in the 'All' universe,
    using whatever prices/fundamentals/membership _capture_pit_data_job
    already landed for today (10 minutes earlier). Own flag, defaults ON
    like the PIT captures it depends on -- internal accumulation, no
    legal/compliance gate."""
    if not await get_setting_bool(STOCK_SCORE_COMPUTE_ENABLED_KEY, default=True):
        logger.info("Scheduler: stock_score_compute is disabled, skipping this run")
        return

    inserted = await compute_and_persist_daily_scores()
    if inserted:
        logger.info("Scheduler: stock score capture — %d tickers newly scored", inserted)


SIGNAL_CHANGE_ALERTS_HOUR_ET = 16
# 3 minutes after STOCK_SCORE_COMPUTE_HOUR_ET/MINUTE_ET (16:15) so today's
# fresh stock_scores row exists to compare against yesterday's.
SIGNAL_CHANGE_ALERTS_MINUTE_ET = 18


async def _scan_signal_change_alerts_job() -> None:
    """ALR-1: day-over-day short_signal/long_signal change, per owned or
    watchlisted ticker, emailed once per (user, ticker, horizon, day). Off
    by default, same opt-in posture as portfolio_drop_alerts/
    saved_screen_alerts -- the first thing that emails a user about a
    plain signal change."""
    if not await get_setting_bool(SIGNAL_CHANGE_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: signal_change_alerts is disabled, skipping this run")
        return
    emailed = await scan_signal_changes()
    if emailed:
        logger.info("Scheduler: %d signal-change alert emails sent", emailed)


# Pre-market, so a user sees "earnings in 2 days" with enough notice
# before that trading day, not after it's already underway. Runs daily
# (not just weekdays) since a Monday run needs to catch a Tuesday
# earnings date the same as any other day.
EARNINGS_ALERTS_HOUR_ET = 8
EARNINGS_ALERTS_MINUTE_ET = 0


async def _scan_earnings_alerts_job() -> None:
    """ALR-1: "earnings in 2 days" for every owned or watchlisted ticker,
    emailed once per (user, ticker, day) the window check hits. Off by
    default, same opt-in posture as the other ALR-1 scan jobs."""
    if not await get_setting_bool(EARNINGS_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: earnings_alerts is disabled, skipping this run")
        return
    emailed = await scan_earnings_in_window()
    if emailed:
        logger.info("Scheduler: %d earnings alert emails sent", emailed)


SAVED_SCREEN_ALERTS_HOUR_ET = 16
# 15 minutes after STOCK_SCORE_COMPUTE_HOUR_ET/MINUTE_ET so a saved screen
# filtering on Short/Long-Term Score or Signal sees today's fresh
# stock_scores rows, not yesterday's.
SAVED_SCREEN_ALERTS_MINUTE_ET = 30


async def _scan_saved_screen_alerts_job() -> None:
    """SCN-3: re-applies every saved screen's own filters to today's fresh
    Stock Finder table and emails the owner on a genuine enter/leave
    change. Off by default -- the first Stock Finder feature that emails a
    user automatically, same admin-opt-in rationale as portfolio drop
    alerts (see SAVED_SCREEN_ALERTS_ENABLED_KEY)."""
    if not await get_setting_bool(SAVED_SCREEN_ALERTS_ENABLED_KEY, default=False):
        logger.info("Scheduler: saved_screen_alerts is disabled, skipping this run")
        return

    emailed = await scan_saved_screens_for_membership_changes()
    if emailed:
        logger.info("Scheduler: %d saved-screen alert emails sent", emailed)


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


async def _compute_market_regime_job() -> None:
    """REG-1/2/3: computes and persists today's regime row (latest date
    only — see services/market_regime_service.py's one-time admin
    backfill endpoint for historical population). Off by default —
    see MARKET_REGIME_ENABLED_KEY's docstring in app_settings.py for why
    this one gets a stricter opt-in posture than the other daily jobs."""
    if not await get_setting_bool(MARKET_REGIME_ENABLED_KEY, default=False):
        logger.info("Scheduler: market_regime is disabled, skipping this run")
        return

    result = await compute_and_persist_daily_regime(backfill=False)
    if result["rows_persisted"]:
        logger.info(
            "Scheduler: market regime — %s confirmed for %s",
            result["latest_regime"], result["latest_as_of_date"],
        )


async def _compute_filing_summaries_job() -> None:
    """SUM-1: real SEC 10-K/10-Q filing summaries for every ticker any
    user holds or has watchlisted (same owned/watchlisted ticker universe
    as web/backend/routers/earnings.py's calendar, but global across all
    users since this is shared per-ticker data). Off by default — see
    FILING_SUMMARIES_ENABLED_KEY's docstring in app_settings.py: this hits
    a real external (SEC EDGAR) API plus LLM cost on a schedule. Tickers
    are processed one at a time (not fanned out in parallel) to stay a
    well-behaved EDGAR citizen under their stated rate limit."""
    if not await get_setting_bool(FILING_SUMMARIES_ENABLED_KEY, default=False):
        logger.info("Scheduler: filing_summaries is disabled, skipping this run")
        return

    async with service_conn() as conn:
        owned_rows = await conn.fetch("SELECT DISTINCT ticker FROM portfolio_positions")
        watchlisted_rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM watchlist_alerts WHERE active AND source IS DISTINCT FROM 'portfolio_auto'"
        )
    tickers = sorted({r["ticker"] for r in owned_rows} | {r["ticker"] for r in watchlisted_rows})
    if not tickers:
        return

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    if not labels:
        logger.info("Scheduler: filing_summaries has no LLM provider configured, skipping this run")
        return
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)

    total_inserted = 0
    for ticker in tickers:
        try:
            total_inserted += await process_new_filings_for_ticker(llms, ticker)
        except Exception as e:
            logger.warning("Scheduler: filing_summaries failed for %s: %s", ticker, e)

    if total_inserted:
        logger.info("Scheduler: filing summaries — %d new filing(s) summarized", total_inserted)


async def _evaluate_quant_signal_outcomes_job() -> None:
    """The live counterpart to the Quant Signal capture above: checks
    every already-captured call old enough to have a real exit price on
    record and evaluates it. Independent of PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY
    — evaluating already-captured PIT data isn't itself a new capture, so
    it keeps running even if capture is paused (same reasoning as
    evaluate_due_signal_outcomes needing publish_signals_enabled only
    because there'd be nothing to evaluate otherwise)."""
    evaluated = await evaluate_due_quant_signal_outcomes()
    if evaluated:
        logger.info("Scheduler: evaluated %d quant signal outcomes", evaluated)


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
    evaluate.

    TRK-2 (docs/stock-analysis-requirements.html): track-record metrics
    are grouped by horizon (10/30/60/90 days), so this now evaluates all
    of TRACK_RECORD_HORIZONS each run, not just the 30-day default —
    signal_outcomes' unique constraint already includes horizon_days, so
    this is purely a matter of calling evaluate_due_signal_outcomes once
    per horizon; each call is independently idempotent."""
    if not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        logger.info("Scheduler: publish_daily_signals is disabled, skipping outcome evaluation")
        return
    total = 0
    for horizon_days in TRACK_RECORD_HORIZONS:
        total += await evaluate_due_signal_outcomes(horizon_days=horizon_days)
    if total:
        logger.info("Scheduler: recorded %d signal outcomes across %d horizons", total, len(TRACK_RECORD_HORIZONS))


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
        _scan_cost_drop_alerts_job,
        "interval",
        minutes=PORTFOLIO_DROP_INTERVAL_MINUTES,
        id="scan_cost_drop_alerts",
        next_run_time=datetime.now(),
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _flush_pending_digest_job,
        "interval",
        minutes=60,
        id="flush_pending_digest",
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _poll_paper_orders_job,
        "interval",
        seconds=PAPER_ORDER_POLL_INTERVAL_SECONDS,
        id="poll_paper_orders",
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _sync_paper_positions_job,
        "interval",
        minutes=PAPER_POSITIONS_SYNC_INTERVAL_MINUTES,
        id="sync_paper_positions",
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
        _compute_stock_scores_job,
        CronTrigger(
            hour=STOCK_SCORE_COMPUTE_HOUR_ET, minute=STOCK_SCORE_COMPUTE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="compute_stock_scores",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _scan_signal_change_alerts_job,
        CronTrigger(
            hour=SIGNAL_CHANGE_ALERTS_HOUR_ET, minute=SIGNAL_CHANGE_ALERTS_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="scan_signal_change_alerts",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _scan_earnings_alerts_job,
        CronTrigger(hour=EARNINGS_ALERTS_HOUR_ET, minute=EARNINGS_ALERTS_MINUTE_ET, timezone="America/New_York"),
        id="scan_earnings_alerts",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _scan_saved_screen_alerts_job,
        CronTrigger(
            hour=SAVED_SCREEN_ALERTS_HOUR_ET, minute=SAVED_SCREEN_ALERTS_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="scan_saved_screen_alerts",
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
        _compute_market_regime_job,
        CronTrigger(
            hour=MARKET_REGIME_HOUR_ET, minute=MARKET_REGIME_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="compute_market_regime",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _compute_filing_summaries_job,
        CronTrigger(
            hour=FILING_SUMMARIES_HOUR_ET, minute=FILING_SUMMARIES_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="compute_filing_summaries",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _evaluate_quant_signal_outcomes_job,
        CronTrigger(
            hour=QUANT_SIGNAL_EVALUATE_HOUR_ET, minute=QUANT_SIGNAL_EVALUATE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="evaluate_quant_signal_outcomes",
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
    _scheduler.add_job(
        _check_basket_rebalances_monthly_job,
        CronTrigger(day="1", hour=BASKET_REBALANCE_HOUR_ET, minute=BASKET_REBALANCE_MINUTE_ET, timezone="America/New_York"),
        id="check_basket_rebalances_monthly",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _check_basket_rebalances_quarterly_job,
        CronTrigger(
            month=BASKET_REBALANCE_QUARTERLY_MONTHS, day="1",
            hour=BASKET_REBALANCE_HOUR_ET, minute=BASKET_REBALANCE_MINUTE_ET,
            timezone="America/New_York",
        ),
        id="check_basket_rebalances_quarterly",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _prewarm_stock_finder_cache_job,
        "interval",
        minutes=STOCK_FINDER_PREWARM_INTERVAL_MINUTES,
        id="prewarm_stock_finder_cache",
        next_run_time=datetime.now(),
        coalesce=True,
        max_instances=1,
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
    logger.info(
        "Scheduler: basket rebalance checks monthly/quarterly %02d:%02d ET (quarterly months %s), "
        "stock-finder cache prewarm every %d min",
        BASKET_REBALANCE_HOUR_ET, BASKET_REBALANCE_MINUTE_ET, BASKET_REBALANCE_QUARTERLY_MONTHS,
        STOCK_FINDER_PREWARM_INTERVAL_MINUTES,
    )
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
