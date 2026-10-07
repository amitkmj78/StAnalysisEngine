from web.backend.db import service_conn

VERIFY_PREDICTIONS_ENABLED_KEY = "verify_predictions_enabled"
# Defaults OFF deliberately: CMP-01/Q-01/Q-02 require counsel confirmation
# that unpaid, impersonal publication carries no registration requirement
# *before* the first real publication happens. Deploying the publication
# pipeline must not itself start publishing — an explicit admin opt-in does.
PUBLISH_SIGNALS_ENABLED_KEY = "publish_signals_enabled"
# Defaults ON — the secure default. Admin can turn it off (e.g. temporarily,
# to debug a signup issue) without a deploy.
PASSWORD_POLICY_ENABLED_KEY = "password_policy_enabled"
# Gates all TR-3 daily capture — prices, universe membership, fundamentals
# (Phases 1-3, one job, one flag). Defaults ON — internal data capture with
# no legal/compliance gate like publish_signals_enabled has. Admin can pause
# it (e.g. yfinance rate limits, a bad run) without losing history already
# captured. Key name kept as-is (pit_price_capture_enabled) since it's
# already live in production — renaming would need a settings migration for
# no functional gain.
PIT_PRICE_CAPTURE_ENABLED_KEY = "pit_price_capture_enabled"
# Same rationale as PIT_PRICE_CAPTURE_ENABLED_KEY (internal data capture,
# no legal gate) — defaults ON. Separate flag since analyst-rating capture
# is cheap (one network call/ticker) and safe to leave running even if the
# quant-signal capture below needs pausing.
PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY = "pit_analyst_rating_capture_enabled"
# Separate from the flag above because this one is expensive — trains a
# model per ticker, ~500 tickers, real CPU load on the same box serving
# live traffic. An admin may want to pause just this one (e.g. during a
# traffic spike) without also pausing the cheap price/fundamentals/
# analyst-rating captures. Defaults ON per explicit request to run daily.
PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY = "pit_quant_signal_capture_enabled"
# Defaults OFF: unlike the other flags above, this triggers a real
# sentiment search (self-hosted, services.web_search) plus an LLM call per
# drop detected, and writes user-visible content — an admin should opt in
# deliberately rather than have it start emailing/notifying users the
# moment this deploys.
PORTFOLIO_DROP_ALERTS_ENABLED_KEY = "portfolio_drop_alerts_enabled"
# The drop-detection threshold itself, admin-configurable — separate from
# the enable/disable flag above so an admin can tune sensitivity without
# a deploy. Stored as text like every other app_settings value; parsed as
# a float on read.
PORTFOLIO_DROP_THRESHOLD_PCT_KEY = "portfolio_drop_threshold_pct"
PORTFOLIO_DROP_THRESHOLD_DEFAULT = 1.0
# Per-user daily request cap enforced by enforce_daily_quota (web/backend/
# rate_limit.py), shared across every quota-gated endpoint. Admin-tunable
# without a deploy — e.g. to raise it temporarily for a user hitting real
# usage, or lower it if something is hammering the API.
DAILY_QUOTA_KEY = "daily_quota"
DAILY_QUOTA_DEFAULT = 600
# NFR-03: defaults ON, like PIT capture — internal safety mechanism, no
# legal/compliance gate. Admin can pause it (e.g. if pg_dump load ever
# becomes a problem) without losing backups already taken.
DB_BACKUP_ENABLED_KEY = "db_backup_enabled"
# Horizon 1 (docs/signal-licensing-whitelabel-requirements.md.pdf, RS-*):
# the paid-subscription layer on top of the existing free/public track
# record. Defaults OFF for the same reason as PUBLISH_SIGNALS_ENABLED_KEY,
# one level further: Gate 0->1 in that spec requires >=6 months of
# continuous live publication (nowhere close yet) AND written counsel
# confirmation that the offering sits within the publisher's exclusion
# (CMP-03). This code is built and testable but must stay off — flipping
# it on is a real business/legal decision, not a deploy.
HORIZON1_SUBSCRIPTIONS_ENABLED_KEY = "horizon1_subscriptions_enabled"
# RS-2: how many days behind "current" the free tier sees once Horizon 1
# is live. Admin-tunable without a deploy, same rationale as
# PORTFOLIO_DROP_THRESHOLD_PCT_KEY.
FREE_TIER_LAG_DAYS_KEY = "free_tier_lag_days"
FREE_TIER_LAG_DAYS_DEFAULT = 7
# Which live-quote source get_latest_price/get_extended_hours_price
# (services/data_service.py) use: "yahoo" (yfinance, unofficial/scraped,
# free) or "alpaca" (Alpaca's free real-time IEX feed, needs
# ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY). Defaults "yahoo" to preserve
# existing behavior. Persisted here (survives a restart) but the value
# actually read at request time is services.price_provider's in-process
# cache — see that module's docstring for why.
PRICE_DATA_PROVIDER_KEY = "price_data_provider"
PRICE_DATA_PROVIDER_DEFAULT = "yahoo"
# Defaults OFF: re-ranks a whole universe per due basket (real yfinance
# load) and writes user-visible content (a rebalance alert), same
# opt-in rationale as PORTFOLIO_DROP_ALERTS_ENABLED_KEY.
BASKET_REBALANCE_ENABLED_KEY = "basket_rebalance_enabled"
# Defaults OFF: re-ranks every saved screen's universe nightly and, on a
# genuine enter/leave change, emails the screen's owner (SCN-3) -- the
# first Stock Finder feature that emails a user automatically, so it gets
# the same deliberate admin opt-in as PORTFOLIO_DROP_ALERTS_ENABLED_KEY
# rather than defaulting on the moment this deploys.
SAVED_SCREEN_ALERTS_ENABLED_KEY = "saved_screen_alerts_enabled"
# Defaults OFF: a continuous ~50-minute-interval scan of the "All"/S&P 500
# stock-finder universes purely to keep get_stock_finder_table's cache
# warm (see the Diversified Basket page's <3s generation goal) — a real,
# ongoing increase in steady-state Yahoo Finance traffic an admin should
# opt into deliberately, not something defaulted on silently.
STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY = "stock_finder_cache_prewarm_enabled"
# Defaults OFF: deploying the paper-trading code must not itself start
# accepting order submissions or scheduler polling — an explicit admin
# opt-in does, same rationale as PORTFOLIO_DROP_ALERTS_ENABLED_KEY.
PAPER_TRADING_ENABLED_KEY = "paper_trading_enabled"
# TRD-35 kill switch. Defaults OFF (= not engaged, submits allowed).
# Checked synchronously at the top of the order-submit endpoint, so
# engaging it blocks new submits immediately — not on the next poll cycle.
PAPER_TRADING_KILL_SWITCH_KEY = "paper_trading_kill_switch"
# TRD-16 per-order value limit, admin-tunable without a deploy.
PAPER_TRADING_MAX_ORDER_VALUE_KEY = "paper_trading_max_order_value"
PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT = 10_000.0
# TRD-16 max % of portfolio equity in one order.
PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY = "paper_trading_max_portfolio_pct"
PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT = 25.0
# TRD-16 max orders per user per day.
PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY = "paper_trading_max_orders_per_day"
PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT = 20
# TRD-17 price-collar warning threshold, as a percent deviation from last
# trade price, for limit orders only.
PAPER_TRADING_PRICE_COLLAR_PCT_KEY = "paper_trading_price_collar_pct"
PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT = 20.0
# TRD-19 restricted-symbol blocklist, comma-separated tickers, admin-set.
# Empty by default — no symbols are restricted until an admin adds some.
PAPER_TRADING_RESTRICTED_SYMBOLS_KEY = "paper_trading_restricted_symbols"
PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT = ""
# Phase 1 ("Trust") two-score system (docs/stock-analysis-requirements.html,
# SCR-1..4): defaults ON, same rationale as PIT_PRICE_CAPTURE_ENABLED_KEY --
# internal data accumulation off the existing PIT stores, no legal/
# compliance gate like publish_signals_enabled has.
STOCK_SCORE_COMPUTE_ENABLED_KEY = "stock_score_compute_enabled"
# REG-1/2/3: gates services/market_regime_service.py's daily scheduler
# job. Defaults OFF, doubly deliberate beyond the usual "deploying code
# must not itself start a live job" rationale — this wires up a scoring
# engine that failed its own release-gate backtest three times (see
# services/market_internals_service.py's module docstring). An admin
# opts in via /admin/settings with that history in view, not by deploy.
MARKET_REGIME_ENABLED_KEY = "market_regime_enabled"
# AGT-21..24: gates the trading agent's AI reviewer step (services/agent/
# reviewer.py), which may only remove a proposed new-entry buy, never add
# or resize one. Defaults OFF, same deliberate-opt-in rationale as
# MARKET_REGIME_ENABLED_KEY -- this is a real LLM call plus an external
# news fetch on every scheduled run, not something deploying code alone
# should turn on.
AI_REVIEWER_ENABLED_KEY = "agent_ai_reviewer_enabled"
# SUM-1: gates services/filing_summary_service.py's daily scheduler job.
# Defaults OFF, same deliberate-opt-in rationale as MARKET_REGIME_ENABLED_
# KEY -- this one hits a real external (SEC EDGAR) API plus LLM cost on a
# schedule, so deploying the code must not itself start making requests.
FILING_SUMMARIES_ENABLED_KEY = "filing_summaries_enabled"
# Signal explanation step 1: gates services/news_ingest_service.py's hourly job that stores SEC 8-K filings as news.
# Defaults OFF, same opt-in rationale: it makes SEC EDGAR requests on a schedule.
NEWS_8K_ENABLED_KEY = "news_8k_enabled"
# SUM-2: gates services/earnings_release_service.py's daily scheduler job.
# Defaults OFF, same rationale as FILING_SUMMARIES_ENABLED_KEY -- also
# shares that job's same daily LLM provider quota, so enabling both at
# once increases the risk of either running out of budget on a given day.
EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY = "earnings_release_summaries_enabled"
# BRF-2: gates services/daily_brief_service.py's build_evening_recap via
# the scheduler's daily per-user job. Defaults OFF, same deliberate-opt-
# in rationale as every other scheduled job -- this emails every user
# with at least one position on a schedule the moment it's deployed
# otherwise. Pure arithmetic (no LLM cost), unlike MORNING_BRIEF_ENABLED_
# KEY below.
EVENING_RECAP_ENABLED_KEY = "evening_recap_enabled"
# BRF-1: gates services/daily_brief_service.py's build_morning_brief via
# the scheduler's daily per-user job. Defaults OFF for the same reason
# as EVENING_RECAP_ENABLED_KEY, plus its own extra risk: the "top news"
# section calls score_tickers_sentiment (LLM cost, capped at 3 calls/
# user/day), sharing the same daily provider quota as Filing/Earnings-
# Release Summaries -- see FILING_SUMMARIES_ENABLED_KEY's docstring for
# the real quota-exhaustion incident this is guarding against.
MORNING_BRIEF_ENABLED_KEY = "morning_brief_enabled"
# PPR-2: gates services/challenge_service.py's capture_equity_for_account
# via the scheduler's daily per-paper-account job. Defaults OFF, same
# deliberate-opt-in rationale as PAPER_TRADING_ENABLED_KEY -- this hits a
# real external (Alpaca) API using each user's own stored paper-trading
# credentials on a schedule. Challenge leaderboards simply show "not
# enough data yet" for every member until an admin turns this on.
PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY = "paper_account_equity_capture_enabled"
# ALR-1: each of the three new alert-scan jobs gets its own deliberate
# opt-in flag, same rationale as PORTFOLIO_DROP_ALERTS_ENABLED_KEY --
# they write user-visible content and send email, so a deploy must not
# itself start alerting anyone. Defaults OFF.
SIGNAL_CHANGE_ALERTS_ENABLED_KEY = "signal_change_alerts_enabled"
EARNINGS_ALERTS_ENABLED_KEY = "earnings_alerts_enabled"
COST_DROP_ALERTS_ENABLED_KEY = "cost_drop_alerts_enabled"
# Separate from PORTFOLIO_DROP_THRESHOLD_PCT_KEY -- that one means "vs.
# yesterday's close", this means "vs. cost basis", different concepts.
COST_DROP_THRESHOLD_PCT_KEY = "cost_drop_threshold_pct"
COST_DROP_THRESHOLD_DEFAULT = 10.0


async def get_setting_bool(key: str, default: bool) -> bool:
    async with service_conn() as conn:
        value = await conn.fetchval("SELECT value FROM app_settings WHERE key = $1", key)
    if value is None:
        return default
    return value.lower() == "true"


async def set_setting_bool(key: str, value: bool) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key, "true" if value else "false",
        )


async def get_setting_float(key: str, default: float) -> float:
    async with service_conn() as conn:
        value = await conn.fetchval("SELECT value FROM app_settings WHERE key = $1", key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


async def set_setting_float(key: str, value: float) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key, str(value),
        )


async def get_setting_str(key: str, default: str) -> str:
    async with service_conn() as conn:
        value = await conn.fetchval("SELECT value FROM app_settings WHERE key = $1", key)
    return value if value is not None else default


async def set_setting_str(key: str, value: str) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key, value,
        )


async def get_setting_int(key: str, default: int) -> int:
    async with service_conn() as conn:
        value = await conn.fetchval("SELECT value FROM app_settings WHERE key = $1", key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


async def set_setting_int(key: str, value: int) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES ($1, $2, now())
            ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()
            """,
            key, str(value),
        )

# Phase 4 trading agent. Global switch for the scheduled run; defaults OFF so
# deploying the code never starts placing orders. Per-user enablement lives in
# agent_user_settings and requires an admin with a compliance reference.
AGENT_ENABLED_KEY = "agent_enabled"
# Global emergency stop for every user's agent (AGT-3). Per-user kills live on
# agent_user_settings.kill_engaged.
AGENT_KILL_SWITCH_KEY = "agent_kill_switch"

# Challenge notifications (daily rank, passed-you, ending reminder, final
# result). Off by default: this emails members on a schedule, so it needs a
# deliberate admin opt-in, same as every other scheduled alert.
CHALLENGE_NOTIFICATIONS_ENABLED_KEY = "challenge_notifications_enabled"
