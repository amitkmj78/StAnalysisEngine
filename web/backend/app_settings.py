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
