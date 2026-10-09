import logging
from datetime import date, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from starlette.concurrency import run_in_threadpool

from services.alert_engine_service import evaluate_alert
from services.basket_rebalance_service import scan_baskets_for_rebalance
from services.cost_drop_alert_service import scan_cost_drops
from services.agent.runner import run_agent_for_user
from services.challenge_notifications import run_challenge_notifications
from services.challenge_service import capture_equity_for_account
from services.daily_brief_service import build_evening_recap, build_morning_brief
from services.earnings_alert_service import scan_earnings_in_window
from services.earnings_release_service import process_new_earnings_releases_for_ticker
from services.email_service import APP_URL, send_admin_alert_email, send_digest_email, send_rankings_email
from services.filing_summary_service import process_new_filings_for_ticker
from services.news_ingest_service import ingest_8k_news_for_ticker
from services.news_summary_service import summarize_pending_8k_news
from services.market_regime_service import compute_and_persist_daily_regime
from services.notification_dispatcher import EASTERN, dispatch_alert, is_within_quiet_hours
from services.prediction_verification_service import verify_prediction
from services.signal_change_alert_service import scan_signal_changes
from services.saved_screen_alert_service import scan_saved_screens_for_membership_changes
from services.signal_publication_service import DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE, TRACK_RECORD_HORIZONS
from services.stock_finder_service import SP500_UNIVERSE_NAME, get_stock_finder_table
from services.stock_score_capture_service import compute_and_persist_daily_scores
from web.backend.admin import ADMIN_EMAIL
from web.backend.community_ideas_eval import evaluate_due_community_ideas
from web.backend.community_model_author import publish_model_ideas_for_today
from web.backend.social_badges import recompute_verified_badges
from web.backend.condition_alerts_eval import evaluate_due_condition_alerts, evaluate_due_intraday_condition_alerts
from web.backend.app_settings import (
    BASKET_REBALANCE_ENABLED_KEY,
    COST_DROP_ALERTS_ENABLED_KEY,
    DB_BACKUP_ENABLED_KEY,
    EARNINGS_ALERTS_ENABLED_KEY,
    EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY,
    EVENING_RECAP_ENABLED_KEY,
    FILING_SUMMARIES_ENABLED_KEY,
    NEWS_8K_ENABLED_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    MARKET_REGIME_ENABLED_KEY,
    MORNING_BRIEF_ENABLED_KEY,
    AGENT_ENABLED_KEY,
    CHALLENGE_NOTIFICATIONS_ENABLED_KEY,
    PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY,
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
    evaluate_due_stock_page_signal_outcomes,
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
# SUM-2: scheduled after SUM-1 so filing summaries get first claim on the
# day's shared LLM provider quota (confirmed this matters: a real backfill
# run this session hit Groq's daily token cap partway through).
EARNINGS_RELEASE_SUMMARIES_HOUR_ET = 21
EARNINGS_RELEASE_SUMMARIES_MINUTE_ET = 0
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
# FND-3: after both pit_prices (16:05) and stock_scores (16:15) capture for
# the day, same "once daily, after its inputs are ready" reasoning as
# EVALUATE_HOUR_ET above -- independent pipeline, just placed in the same
# post-close window.
STOCK_PAGE_EVALUATE_HOUR_ET = 17
STOCK_PAGE_EVALUATE_MINUTE_ET = 5
# COM-7: right after stock_scores capture (16:15) so a fresh signal
# change has something to read.
COMMUNITY_MODEL_AUTHOR_HOUR_ET = 16
COMMUNITY_MODEL_AUTHOR_MINUTE_ET = 25
# COM-2: once daily, after its own ideas' entry-day price data is
# settled -- same "once daily, due is measured in trading days, not
# minutes" reasoning as the other evaluate-at-horizon jobs.
COMMUNITY_IDEAS_EVALUATE_HOUR_ET = 17
COMMUNITY_IDEAS_EVALUATE_MINUTE_ET = 10
# SOC-1: right after COMMUNITY_IDEAS_EVALUATE so today's freshly scored
# ideas (if any) are already on record before the badge recompute reads
# the leaderboard.
SOCIAL_VERIFIED_BADGES_HOUR_ET = 17
SOCIAL_VERIFIED_BADGES_MINUTE_ET = 20
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


async def _evaluate_condition_alerts_job() -> None:
    """ALX-1: checks every active, not-yet-triggered multi-condition
    alert. Same interval as _evaluate_watchlist_alerts above for now --
    ALX-2 adds a separate, much tighter interval specifically for alerts
    whose conditions are price/indicator-only (no score/signal/regime/
    earnings term), since those are the only ones a faster poll can
    actually move the needle on."""
    triggered = await evaluate_due_condition_alerts()
    if triggered:
        logger.info("Scheduler: triggered %d condition alerts", triggered)


async def _evaluate_intraday_condition_alerts_job() -> None:
    """ALX-2: the much tighter interval, restricted (inside
    evaluate_due_intraday_condition_alerts itself) to alerts whose
    conditions are entirely price/indicator fields -- see that
    function's own docstring for why score/signal/regime/earnings
    conditions are excluded here."""
    triggered = await evaluate_due_intraday_condition_alerts()
    if triggered:
        logger.info("Scheduler: triggered %d intraday condition alerts", triggered)


async def _publish_community_model_ideas_job() -> None:
    """COM-7: the app's own model appears as its own leaderboard author
    -- reuses signal_change_alert_service.py's existing fresh-signal-
    change detection, scoped to the whole universe."""
    published = await publish_model_ideas_for_today()
    if published:
        logger.info("Scheduler: published %d model-authored community ideas", published)


async def _evaluate_community_ideas_job() -> None:
    """COM-2: scores every unscored community idea whose horizon has
    elapsed, against SPY -- see web/backend/community_ideas_eval.py."""
    scored = await evaluate_due_community_ideas()
    if scored:
        logger.info("Scheduler: scored %d community ideas", scored)


async def _recompute_verified_badges_job() -> None:
    """SOC-1: recomputes every user's verified_badge from their real
    COM-3/COM-4 scored-idea track record -- never user-settable. See
    web/backend/social_badges.py."""
    changed = await recompute_verified_badges()
    if changed:
        logger.info("Scheduler: verified_badge changed for %d user(s)", changed)


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


# 15:45 ET: ten minutes before the close, so the run sees the day's
# regime and scores and any order still has a session to fill in.
TRADING_AGENT_HOUR_ET = 15
TRADING_AGENT_MINUTE_ET = 45


async def _run_trading_agents_job() -> None:
    """Phase 4: runs the trading agent for every user an admin has enabled.
    Off by default (AGENT_ENABLED_KEY). The runner itself refuses to place
    orders outside paper mode and while the broker's market is closed.
    One user's failure is isolated and never stops the rest."""
    if not await get_setting_bool(AGENT_ENABLED_KEY, default=False):
        logger.info("Scheduler: trading agent is disabled, skipping this run")
        return
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT user_id FROM agent_user_settings WHERE enabled")
    for row in rows:
        try:
            result = await run_agent_for_user(str(row["user_id"]), trigger="scheduled")
            logger.info("Scheduler: trading agent for user %s -> %s", row["user_id"], result.get("status"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Scheduler: trading agent failed for user %s: %s", row["user_id"], e)


# After the 16:20 ET equity capture, so today's snapshots feed the ranks.
CHALLENGE_NOTIFICATIONS_HOUR_ET = 16
CHALLENGE_NOTIFICATIONS_MINUTE_ET = 40


async def _send_challenge_notifications_job() -> None:
    """Challenge rank emails and alerts. Off by default (CHALLENGE_NOTIFICATIONS_
    ENABLED_KEY). Every message is claimed once per member, kind and day, so a
    rerun cannot double-send."""
    if not await get_setting_bool(CHALLENGE_NOTIFICATIONS_ENABLED_KEY, default=False):
        logger.info("Scheduler: challenge notifications are disabled, skipping this run")
        return
    sent = await run_challenge_notifications()
    if sent:
        logger.info("Scheduler: sent %d challenge notification(s)", sent)

# After close (16:00 ET) with a few minutes' buffer, same spacing
# rationale as the other post-close captures already scheduled here.
PAPER_ACCOUNT_EQUITY_CAPTURE_HOUR_ET = 16
PAPER_ACCOUNT_EQUITY_CAPTURE_MINUTE_ET = 20


async def _capture_paper_account_equity_job() -> None:
    """PPR-2: one equity snapshot per linked paper-trading account per
    day -- the return series a challenge leaderboard needs to show risk,
    not just a single live balance. Off by default, its own flag
    separate from PAPER_TRADING_ENABLED_KEY (see PAPER_ACCOUNT_EQUITY_
    CAPTURE_ENABLED_KEY's docstring in app_settings.py) since this is a
    distinct feature's data capture, not paper-trading's own order flow.
    Mirrors sync_positions_and_cash's exact loop shape (service_conn()
    fetch of every active account, per-account try/except already
    inside capture_equity_for_account so one bad key doesn't stop the
    rest)."""
    if not await get_setting_bool(PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY, default=False):
        logger.info("Scheduler: paper_account_equity_capture is disabled, skipping this run")
        return
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM alpaca_paper_accounts WHERE status = 'active'")
    captured = 0
    for row in rows:
        try:
            if await capture_equity_for_account(dict(row)):
                captured += 1
        except Exception as e:
            logger.warning("Scheduler: paper_account_equity_capture failed for account %s: %s", row["id"], e)
    if captured:
        logger.info("Scheduler: captured %d paper-account equity snapshot(s)", captured)


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


async def _refresh_news_8k_job() -> None:
    """Signal explanation step 1: store the last 30 days of SEC 8-Ks for portfolio and watchlist tickers, one at a time
    (same polite pace as the filing summary job). Off by default; see NEWS_8K_ENABLED_KEY."""
    if not await get_setting_bool(NEWS_8K_ENABLED_KEY, default=False):
        logger.info("Scheduler: news_8k is disabled, skipping this run")
        return

    async with service_conn() as conn:
        owned_rows = await conn.fetch("SELECT DISTINCT ticker FROM portfolio_positions")
        watchlisted_rows = await conn.fetch(
            "SELECT DISTINCT ticker FROM watchlist_alerts WHERE active AND source IS DISTINCT FROM 'portfolio_auto'"
        )
    tickers = sorted({r["ticker"] for r in owned_rows} | {r["ticker"] for r in watchlisted_rows})
    for ticker in tickers:
        try:
            await ingest_8k_news_for_ticker(ticker)
        except Exception as e:
            logger.warning("Scheduler: news_8k failed for %s: %s", ticker, e)

    # Option 2: summarize a few new non-earnings 8-Ks per run, so LLM cost stays small. Skipped if no LLM is configured.
    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    if labels:
        llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
        try:
            stored = await summarize_pending_8k_news(llms, limit=5)
            if stored:
                logger.info("Scheduler: news_8k — %d filing summary(ies) stored", stored)
        except Exception as e:
            logger.warning("Scheduler: news_8k summaries failed: %s", e)


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


async def _compute_earnings_release_summaries_job() -> None:
    """SUM-2: real earnings press-release summaries (SEC EDGAR 8-K Exhibit
    99.1 -- not a call transcript, see services/earnings_release_service.py's
    NO_QA_CAVEAT), same owned/watchlisted ticker universe as the filing-
    summaries job above. Off by default, and scheduled after it (see
    EARNINGS_RELEASE_SUMMARIES_HOUR_ET/MINUTE_ET) so filing summaries get
    first claim on the day's shared LLM provider quota."""
    if not await get_setting_bool(EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY, default=False):
        logger.info("Scheduler: earnings_release_summaries is disabled, skipping this run")
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
        logger.info("Scheduler: earnings_release_summaries has no LLM provider configured, skipping this run")
        return
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)

    total_inserted = 0
    for ticker in tickers:
        try:
            total_inserted += await process_new_earnings_releases_for_ticker(llms, ticker)
        except Exception as e:
            logger.warning("Scheduler: earnings_release_summaries failed for %s: %s", ticker, e)

    if total_inserted:
        logger.info("Scheduler: earnings release summaries — %d new release(s) summarized", total_inserted)


# Before the 9:30 ET open, with enough notice to actually read it first.
MORNING_BRIEF_HOUR_ET = 7
MORNING_BRIEF_MINUTE_ET = 0


async def _send_morning_briefs_job() -> None:
    """BRF-1: five-section morning brief (overnight moves, signal
    changes, earnings today, market regime, top news) emailed to every
    user with at least one position in an active portfolio. Off by
    default — see MORNING_BRIEF_ENABLED_KEY's docstring in app_settings.py:
    unlike evening_recap, this one does make LLM calls (top news), capped
    at 3/user/day via daily_brief_service.MAX_TOP_NEWS_TICKERS, sharing
    the same daily provider quota as Filing/Earnings-Release Summaries.
    Routed through notification_dispatcher.dispatch_alert like every
    other alert, same as evening_recap."""
    if not await get_setting_bool(MORNING_BRIEF_ENABLED_KEY, default=False):
        logger.info("Scheduler: morning_brief is disabled, skipping this run")
        return

    async with service_conn() as conn:
        position_rows = await conn.fetch(
            """
            SELECT pp.user_id, pp.ticker, pp.shares, pp.avg_cost, pp.acquired_at
            FROM portfolio_positions pp
            JOIN portfolios p ON p.id = pp.portfolio_id
            WHERE p.is_active AND pp.ticker IS NOT NULL AND pp.shares > 0
            """
        )
        watchlist_rows = await conn.fetch(
            "SELECT user_id, ticker FROM watchlist_alerts WHERE active AND source IS DISTINCT FROM 'portfolio_auto'"
        )
    if not position_rows:
        return

    positions_by_user: dict = {}
    for row in position_rows:
        positions_by_user.setdefault(row["user_id"], []).append(
            {
                "ticker": row["ticker"],
                "shares": row["shares"],
                "avg_cost": row["avg_cost"],
                "acquired_at": row["acquired_at"],
            }
        )
    watchlisted_by_user: dict = {}
    for row in watchlist_rows:
        watchlisted_by_user.setdefault(row["user_id"], []).append(row["ticker"])

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels) if labels else []
    if not labels:
        logger.info("Scheduler: morning_brief has no LLM provider configured, top news will be skipped")

    dispatched = 0
    for user_id, positions in positions_by_user.items():
        try:
            brief = await build_morning_brief(llms, str(user_id), positions, watchlisted_by_user.get(user_id, []))
        except Exception as e:
            logger.warning("Scheduler: morning_brief failed for user %s: %s", user_id, e)
            continue
        if brief is None:
            continue
        await dispatch_alert(str(user_id), None, "morning_brief", brief["subject"], brief["text_body"])
        dispatched += 1

    if dispatched:
        logger.info("Scheduler: morning brief — %d email(s) dispatched", dispatched)


# After the regular session's 16:00 ET close, with a few minutes for the
# close itself and get_previous_close/get_effective_price to settle.
EVENING_RECAP_HOUR_ET = 16
EVENING_RECAP_MINUTE_ET = 30


async def _send_evening_recaps_job() -> None:
    """BRF-2: today's portfolio move vs. SPY plus contribution by
    holding, emailed to every user with at least one position in an
    active portfolio. Off by default — see EVENING_RECAP_ENABLED_KEY's
    docstring in app_settings.py. Pure arithmetic (compute_portfolio_
    performance), no LLM cost. Routed through notification_dispatcher.
    dispatch_alert (ticker=None, alert_type="evening_recap") rather than
    emailing directly, so quiet hours/digest/channel preference apply
    the same as any other alert."""
    if not await get_setting_bool(EVENING_RECAP_ENABLED_KEY, default=False):
        logger.info("Scheduler: evening_recap is disabled, skipping this run")
        return

    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT pp.user_id, pp.ticker, pp.shares, pp.avg_cost, pp.acquired_at
            FROM portfolio_positions pp
            JOIN portfolios p ON p.id = pp.portfolio_id
            WHERE p.is_active AND pp.ticker IS NOT NULL AND pp.shares > 0
            """
        )
    if not rows:
        return

    positions_by_user: dict = {}
    for row in rows:
        positions_by_user.setdefault(row["user_id"], []).append(
            {
                "ticker": row["ticker"],
                "shares": row["shares"],
                "avg_cost": row["avg_cost"],
                "acquired_at": row["acquired_at"],
            }
        )

    dispatched = 0
    for user_id, positions in positions_by_user.items():
        try:
            recap = await run_in_threadpool(build_evening_recap, positions)
        except Exception as e:
            logger.warning("Scheduler: evening_recap failed for user %s: %s", user_id, e)
            continue
        if recap is None:
            continue
        await dispatch_alert(str(user_id), None, "evening_recap", recap["subject"], recap["text_body"])
        dispatched += 1

    if dispatched:
        logger.info("Scheduler: evening recap — %d email(s) dispatched", dispatched)


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


async def _evaluate_stock_page_signal_outcomes_job() -> None:
    """FND-3: the real-signal counterpart to _evaluate_signal_outcomes_job
    above. Not gated by PUBLISH_SIGNALS_ENABLED_KEY -- stock_scores is
    captured by the core nightly scoring job regardless of whether the
    separate momentum-publication pipeline is on, so there's always
    something to evaluate here as soon as signals have matured."""
    evaluated = await evaluate_due_stock_page_signal_outcomes()
    if evaluated:
        logger.info("Scheduler: recorded %d stock-page signal outcomes", evaluated)


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
        _evaluate_condition_alerts_job,
        "interval",
        minutes=ALERT_INTERVAL_MINUTES,
        id="evaluate_condition_alerts",
        next_run_time=datetime.now(),
        coalesce=True,
        max_instances=1,
    )
    _scheduler.add_job(
        _evaluate_intraday_condition_alerts_job,
        CronTrigger(
            # A safe superset of 9:30am-4:00pm ET market hours, every
            # minute -- precise open/close timing isn't worth the extra
            # complexity here (a fetch a few minutes outside the real
            # session just re-reads the latest already-available bar,
            # harmlessly).
            hour="9-16", minute="*", day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="evaluate_intraday_condition_alerts",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=55,
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
        _refresh_news_8k_job,
        CronTrigger(minute=15),
        id="refresh_news_8k",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=1800,
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
        _compute_earnings_release_summaries_job,
        CronTrigger(
            hour=EARNINGS_RELEASE_SUMMARIES_HOUR_ET, minute=EARNINGS_RELEASE_SUMMARIES_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="compute_earnings_release_summaries",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _send_morning_briefs_job,
        CronTrigger(
            hour=MORNING_BRIEF_HOUR_ET, minute=MORNING_BRIEF_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="send_morning_briefs",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _send_evening_recaps_job,
        CronTrigger(
            hour=EVENING_RECAP_HOUR_ET, minute=EVENING_RECAP_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="send_evening_recaps",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _send_challenge_notifications_job,
        CronTrigger(
            hour=CHALLENGE_NOTIFICATIONS_HOUR_ET, minute=CHALLENGE_NOTIFICATIONS_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="send_challenge_notifications",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_trading_agents_job,
        CronTrigger(
            hour=TRADING_AGENT_HOUR_ET, minute=TRADING_AGENT_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="run_trading_agents",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=600,
    )
    _scheduler.add_job(
        _capture_paper_account_equity_job,
        CronTrigger(
            hour=PAPER_ACCOUNT_EQUITY_CAPTURE_HOUR_ET, minute=PAPER_ACCOUNT_EQUITY_CAPTURE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="capture_paper_account_equity",
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
        _evaluate_stock_page_signal_outcomes_job,
        CronTrigger(
            hour=STOCK_PAGE_EVALUATE_HOUR_ET, minute=STOCK_PAGE_EVALUATE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="evaluate_stock_page_signal_outcomes",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _publish_community_model_ideas_job,
        CronTrigger(
            hour=COMMUNITY_MODEL_AUTHOR_HOUR_ET, minute=COMMUNITY_MODEL_AUTHOR_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="publish_community_model_ideas",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _evaluate_community_ideas_job,
        CronTrigger(
            hour=COMMUNITY_IDEAS_EVALUATE_HOUR_ET, minute=COMMUNITY_IDEAS_EVALUATE_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="evaluate_community_ideas",
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _recompute_verified_badges_job,
        CronTrigger(
            hour=SOCIAL_VERIFIED_BADGES_HOUR_ET, minute=SOCIAL_VERIFIED_BADGES_MINUTE_ET,
            day_of_week="mon-fri", timezone="America/New_York",
        ),
        id="recompute_verified_badges",
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
