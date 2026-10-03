from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.price_provider import PRICE_PROVIDERS, set_price_provider
from web.backend.admin import require_admin
from web.backend.app_settings import (
    BASKET_REBALANCE_ENABLED_KEY,
    DAILY_QUOTA_DEFAULT,
    DAILY_QUOTA_KEY,
    DB_BACKUP_ENABLED_KEY,
    EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY,
    EVENING_RECAP_ENABLED_KEY,
    FILING_SUMMARIES_ENABLED_KEY,
    FREE_TIER_LAG_DAYS_DEFAULT,
    FREE_TIER_LAG_DAYS_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    MARKET_REGIME_ENABLED_KEY,
    MORNING_BRIEF_ENABLED_KEY,
    PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY,
    CHALLENGE_NOTIFICATIONS_ENABLED_KEY,
    PAPER_TRADING_ENABLED_KEY,
    PAPER_TRADING_KILL_SWITCH_KEY,
    PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT,
    PAPER_TRADING_MAX_ORDER_VALUE_KEY,
    PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT,
    PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY,
    PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT,
    PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY,
    PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT,
    PAPER_TRADING_PRICE_COLLAR_PCT_KEY,
    PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT,
    PAPER_TRADING_RESTRICTED_SYMBOLS_KEY,
    PASSWORD_POLICY_ENABLED_KEY,
    PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY,
    PIT_PRICE_CAPTURE_ENABLED_KEY,
    PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY,
    PORTFOLIO_DROP_ALERTS_ENABLED_KEY,
    PORTFOLIO_DROP_THRESHOLD_DEFAULT,
    PORTFOLIO_DROP_THRESHOLD_PCT_KEY,
    PRICE_DATA_PROVIDER_DEFAULT,
    PRICE_DATA_PROVIDER_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    SAVED_SCREEN_ALERTS_ENABLED_KEY,
    STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY,
    STOCK_SCORE_COMPUTE_ENABLED_KEY,
    VERIFY_PREDICTIONS_ENABLED_KEY,
    get_setting_bool,
    get_setting_float,
    get_setting_int,
    get_setting_str,
    set_setting_bool,
    set_setting_float,
    set_setting_int,
    set_setting_str,
)

router = APIRouter(
    prefix="/api/v1/admin/settings",
    tags=["admin-settings"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
async def get_settings():
    return {
        "verify_predictions_enabled": await get_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, default=True),
        "publish_signals_enabled": await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False),
        "password_policy_enabled": await get_setting_bool(PASSWORD_POLICY_ENABLED_KEY, default=True),
        "pit_price_capture_enabled": await get_setting_bool(PIT_PRICE_CAPTURE_ENABLED_KEY, default=True),
        "pit_analyst_rating_capture_enabled": await get_setting_bool(
            PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY, default=True
        ),
        "pit_quant_signal_capture_enabled": await get_setting_bool(
            PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY, default=True
        ),
        "portfolio_drop_alerts_enabled": await get_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, default=False),
        "portfolio_drop_threshold_pct": await get_setting_float(
            PORTFOLIO_DROP_THRESHOLD_PCT_KEY, default=PORTFOLIO_DROP_THRESHOLD_DEFAULT
        ),
        "daily_quota": await get_setting_int(DAILY_QUOTA_KEY, default=DAILY_QUOTA_DEFAULT),
        "db_backup_enabled": await get_setting_bool(DB_BACKUP_ENABLED_KEY, default=True),
        "horizon1_subscriptions_enabled": await get_setting_bool(HORIZON1_SUBSCRIPTIONS_ENABLED_KEY, default=False),
        "free_tier_lag_days": await get_setting_int(FREE_TIER_LAG_DAYS_KEY, default=FREE_TIER_LAG_DAYS_DEFAULT),
        "price_data_provider": await get_setting_str(PRICE_DATA_PROVIDER_KEY, default=PRICE_DATA_PROVIDER_DEFAULT),
        "basket_rebalance_enabled": await get_setting_bool(BASKET_REBALANCE_ENABLED_KEY, default=False),
        "saved_screen_alerts_enabled": await get_setting_bool(SAVED_SCREEN_ALERTS_ENABLED_KEY, default=False),
        "stock_finder_cache_prewarm_enabled": await get_setting_bool(
            STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY, default=False
        ),
        "paper_trading_enabled": await get_setting_bool(PAPER_TRADING_ENABLED_KEY, default=False),
        "paper_trading_kill_switch": await get_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, default=False),
        "paper_trading_max_order_value": await get_setting_float(
            PAPER_TRADING_MAX_ORDER_VALUE_KEY, default=PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT
        ),
        "paper_trading_max_portfolio_pct": await get_setting_float(
            PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY, default=PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT
        ),
        "paper_trading_max_orders_per_day": await get_setting_int(
            PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY, default=PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT
        ),
        "paper_trading_price_collar_pct": await get_setting_float(
            PAPER_TRADING_PRICE_COLLAR_PCT_KEY, default=PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT
        ),
        "paper_trading_restricted_symbols": await get_setting_str(
            PAPER_TRADING_RESTRICTED_SYMBOLS_KEY, default=PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT
        ),
        "stock_score_compute_enabled": await get_setting_bool(STOCK_SCORE_COMPUTE_ENABLED_KEY, default=True),
        "market_regime_enabled": await get_setting_bool(MARKET_REGIME_ENABLED_KEY, default=False),
        "filing_summaries_enabled": await get_setting_bool(FILING_SUMMARIES_ENABLED_KEY, default=False),
        "earnings_release_summaries_enabled": await get_setting_bool(
            EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY, default=False
        ),
        "evening_recap_enabled": await get_setting_bool(EVENING_RECAP_ENABLED_KEY, default=False),
        "morning_brief_enabled": await get_setting_bool(MORNING_BRIEF_ENABLED_KEY, default=False),
        "challenge_notifications_enabled": await get_setting_bool(CHALLENGE_NOTIFICATIONS_ENABLED_KEY, default=False),
        "paper_account_equity_capture_enabled": await get_setting_bool(
            PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY, default=False
        ),
    }


@router.post("/verify-predictions/enable")
async def enable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, True)
    return {"verify_predictions_enabled": True}


@router.post("/verify-predictions/disable")
async def disable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, False)
    return {"verify_predictions_enabled": False}


@router.post("/market-regime/enable")
async def enable_market_regime():
    """REG-1: turns on the daily scheduler job (services/market_regime_
    service.py via _compute_market_regime_job) -- see MARKET_REGIME_
    ENABLED_KEY's docstring in app_settings.py for why this defaults off
    and needs a deliberate admin opt-in."""
    await set_setting_bool(MARKET_REGIME_ENABLED_KEY, True)
    return {"market_regime_enabled": True}


@router.post("/market-regime/disable")
async def disable_market_regime():
    await set_setting_bool(MARKET_REGIME_ENABLED_KEY, False)
    return {"market_regime_enabled": False}


@router.post("/filing-summaries/enable")
async def enable_filing_summaries():
    """SUM-1: turns on the daily scheduler job (services/filing_summary_
    service.py via _compute_filing_summaries_job) -- see FILING_SUMMARIES_
    ENABLED_KEY's docstring in app_settings.py for why this defaults off
    and needs a deliberate admin opt-in (hits a real external API plus
    LLM cost)."""
    await set_setting_bool(FILING_SUMMARIES_ENABLED_KEY, True)
    return {"filing_summaries_enabled": True}


@router.post("/filing-summaries/disable")
async def disable_filing_summaries():
    await set_setting_bool(FILING_SUMMARIES_ENABLED_KEY, False)
    return {"filing_summaries_enabled": False}


@router.post("/earnings-release-summaries/enable")
async def enable_earnings_release_summaries():
    """SUM-2: turns on the daily scheduler job (services/earnings_release_
    service.py via _compute_earnings_release_summaries_job) -- see
    EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY's docstring in app_settings.py
    for why this defaults off and shares the filing-summaries job's LLM
    quota."""
    await set_setting_bool(EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY, True)
    return {"earnings_release_summaries_enabled": True}


@router.post("/earnings-release-summaries/disable")
async def disable_earnings_release_summaries():
    await set_setting_bool(EARNINGS_RELEASE_SUMMARIES_ENABLED_KEY, False)
    return {"earnings_release_summaries_enabled": False}


@router.post("/evening-recap/enable")
async def enable_evening_recap():
    """BRF-2: turns on the daily (weekdays 16:30 ET) per-user evening
    recap job (services/daily_brief_service.py via _send_evening_recaps_
    job) -- see EVENING_RECAP_ENABLED_KEY's docstring in app_settings.py
    for why this defaults off and needs a deliberate admin opt-in (emails
    every user with a position)."""
    await set_setting_bool(EVENING_RECAP_ENABLED_KEY, True)
    return {"evening_recap_enabled": True}


@router.post("/evening-recap/disable")
async def disable_evening_recap():
    await set_setting_bool(EVENING_RECAP_ENABLED_KEY, False)
    return {"evening_recap_enabled": False}


@router.post("/morning-brief/enable")
async def enable_morning_brief():
    """BRF-1: turns on the daily (weekdays 07:00 ET) per-user morning
    brief job (services/daily_brief_service.py via _send_morning_briefs_
    job) -- see MORNING_BRIEF_ENABLED_KEY's docstring in app_settings.py
    for why this defaults off: it makes real LLM calls (top news, capped
    at 3/user/day) sharing the same daily quota as Filing/Earnings-
    Release Summaries, on top of the usual "emails every user" opt-in
    rationale."""
    await set_setting_bool(MORNING_BRIEF_ENABLED_KEY, True)
    return {"morning_brief_enabled": True}


@router.post("/morning-brief/disable")
async def disable_morning_brief():
    await set_setting_bool(MORNING_BRIEF_ENABLED_KEY, False)
    return {"morning_brief_enabled": False}


@router.post("/challenge-notifications/enable")
async def enable_challenge_notifications():
    """Turns on the daily (weekdays 16:40 ET) challenge rank emails and alerts."""
    await set_setting_bool(CHALLENGE_NOTIFICATIONS_ENABLED_KEY, True)
    return {"challenge_notifications_enabled": True}


@router.post("/challenge-notifications/disable")
async def disable_challenge_notifications():
    await set_setting_bool(CHALLENGE_NOTIFICATIONS_ENABLED_KEY, False)
    return {"challenge_notifications_enabled": False}


@router.post("/paper-account-equity-capture/enable")
async def enable_paper_account_equity_capture():
    """PPR-2: turns on the daily (weekdays 16:20 ET) per-paper-account
    equity snapshot job (services/challenge_service.py via
    _capture_paper_account_equity_job) -- see PAPER_ACCOUNT_EQUITY_
    CAPTURE_ENABLED_KEY's docstring in app_settings.py for why this
    defaults off (hits live Alpaca credentials). Challenge leaderboards
    show "not enough data yet" for every member until this is on."""
    await set_setting_bool(PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY, True)
    return {"paper_account_equity_capture_enabled": True}


@router.post("/paper-account-equity-capture/disable")
async def disable_paper_account_equity_capture():
    await set_setting_bool(PAPER_ACCOUNT_EQUITY_CAPTURE_ENABLED_KEY, False)
    return {"paper_account_equity_capture_enabled": False}


@router.post("/publish-signals/enable")
async def enable_publish_signals():
    """Flip on only after CMP-01/Q-01/Q-02 are cleared — this starts the
    real, irreversible public track record (TR-1's daily scheduled job)."""
    await set_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, True)
    return {"publish_signals_enabled": True}


@router.post("/publish-signals/disable")
async def disable_publish_signals():
    await set_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, False)
    return {"publish_signals_enabled": False}


@router.post("/password-policy/enable")
async def enable_password_policy():
    await set_setting_bool(PASSWORD_POLICY_ENABLED_KEY, True)
    return {"password_policy_enabled": True}


@router.post("/password-policy/disable")
async def disable_password_policy():
    """Signup still enforces an 8-character floor even when disabled —
    this only turns off the length/complexity/breach-list requirements,
    never lets through a password of any length."""
    await set_setting_bool(PASSWORD_POLICY_ENABLED_KEY, False)
    return {"password_policy_enabled": False}


@router.post("/pit-price-capture/enable")
async def enable_pit_price_capture():
    await set_setting_bool(PIT_PRICE_CAPTURE_ENABLED_KEY, True)
    return {"pit_price_capture_enabled": True}


@router.post("/pit-price-capture/disable")
async def disable_pit_price_capture():
    await set_setting_bool(PIT_PRICE_CAPTURE_ENABLED_KEY, False)
    return {"pit_price_capture_enabled": False}


@router.post("/pit-analyst-rating-capture/enable")
async def enable_pit_analyst_rating_capture():
    await set_setting_bool(PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY, True)
    return {"pit_analyst_rating_capture_enabled": True}


@router.post("/pit-analyst-rating-capture/disable")
async def disable_pit_analyst_rating_capture():
    await set_setting_bool(PIT_ANALYST_RATING_CAPTURE_ENABLED_KEY, False)
    return {"pit_analyst_rating_capture_enabled": False}


@router.post("/pit-quant-signal-capture/enable")
async def enable_pit_quant_signal_capture():
    await set_setting_bool(PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY, True)
    return {"pit_quant_signal_capture_enabled": True}


@router.post("/pit-quant-signal-capture/disable")
async def disable_pit_quant_signal_capture():
    await set_setting_bool(PIT_QUANT_SIGNAL_CAPTURE_ENABLED_KEY, False)
    return {"pit_quant_signal_capture_enabled": False}


@router.post("/portfolio-drop-alerts/enable")
async def enable_portfolio_drop_alerts():
    """Starts real per-drop sentiment search + LLM calls and user-visible
    in-app notifications on the next scheduler tick — not just a preview
    toggle."""
    await set_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, True)
    return {"portfolio_drop_alerts_enabled": True}


@router.post("/portfolio-drop-alerts/disable")
async def disable_portfolio_drop_alerts():
    await set_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, False)
    return {"portfolio_drop_alerts_enabled": False}


@router.post("/basket-rebalance/enable")
async def enable_basket_rebalance():
    """Starts the monthly/quarterly re-rank + drift-check job on the next
    scheduler tick — writes review-and-act alerts, never executes a
    trade itself."""
    await set_setting_bool(BASKET_REBALANCE_ENABLED_KEY, True)
    return {"basket_rebalance_enabled": True}


@router.post("/basket-rebalance/disable")
async def disable_basket_rebalance():
    await set_setting_bool(BASKET_REBALANCE_ENABLED_KEY, False)
    return {"basket_rebalance_enabled": False}


@router.post("/saved-screen-alerts/enable")
async def enable_saved_screen_alerts():
    """Starts the nightly saved-screen membership scan on the next
    scheduler tick -- emails a screen's owner only on a genuine
    enter/leave change, never on a no-op re-run."""
    await set_setting_bool(SAVED_SCREEN_ALERTS_ENABLED_KEY, True)
    return {"saved_screen_alerts_enabled": True}


@router.post("/saved-screen-alerts/disable")
async def disable_saved_screen_alerts():
    await set_setting_bool(SAVED_SCREEN_ALERTS_ENABLED_KEY, False)
    return {"saved_screen_alerts_enabled": False}


@router.post("/stock-finder-cache-prewarm/enable")
async def enable_stock_finder_cache_prewarm():
    """Starts a continuous ~50-minute-interval scan of the 'All'/S&P 500
    universes purely to keep get_stock_finder_table's cache warm — a real,
    ongoing increase in steady-state Yahoo Finance traffic."""
    await set_setting_bool(STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY, True)
    return {"stock_finder_cache_prewarm_enabled": True}


@router.post("/stock-finder-cache-prewarm/disable")
async def disable_stock_finder_cache_prewarm():
    await set_setting_bool(STOCK_FINDER_CACHE_PREWARM_ENABLED_KEY, False)
    return {"stock_finder_cache_prewarm_enabled": False}


class ThresholdUpdate(BaseModel):
    threshold_pct: float = Field(gt=0, le=50)


@router.post("/portfolio-drop-alerts/threshold")
async def set_portfolio_drop_threshold(body: ThresholdUpdate):
    """Takes effect on the next scheduler tick (every 15 min) or the next
    manual Scan Now — no restart needed, read fresh from app_settings each run."""
    await set_setting_float(PORTFOLIO_DROP_THRESHOLD_PCT_KEY, body.threshold_pct)
    return {"portfolio_drop_threshold_pct": body.threshold_pct}


class DailyQuotaUpdate(BaseModel):
    daily_quota: int = Field(gt=0, le=100000)


@router.post("/daily-quota")
async def set_daily_quota(body: DailyQuotaUpdate):
    """Applies to the very next quota-gated request — enforce_daily_quota
    reads this fresh on every call, no restart needed."""
    await set_setting_int(DAILY_QUOTA_KEY, body.daily_quota)
    return {"daily_quota": body.daily_quota}


@router.post("/db-backup/enable")
async def enable_db_backup():
    await set_setting_bool(DB_BACKUP_ENABLED_KEY, True)
    return {"db_backup_enabled": True}


@router.post("/db-backup/disable")
async def disable_db_backup():
    await set_setting_bool(DB_BACKUP_ENABLED_KEY, False)
    return {"db_backup_enabled": False}


@router.post("/horizon1-subscriptions/enable")
async def enable_horizon1_subscriptions():
    """Do not flip this on without written counsel confirmation (Gate 0->1,
    CMP-03) and >=6 months of continuous live publication — see
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY's comment in app_settings.py. This
    endpoint doesn't and can't verify either of those; it's a raw switch."""
    await set_setting_bool(HORIZON1_SUBSCRIPTIONS_ENABLED_KEY, True)
    return {"horizon1_subscriptions_enabled": True}


@router.post("/horizon1-subscriptions/disable")
async def disable_horizon1_subscriptions():
    await set_setting_bool(HORIZON1_SUBSCRIPTIONS_ENABLED_KEY, False)
    return {"horizon1_subscriptions_enabled": False}


class FreeTierLagDaysUpdate(BaseModel):
    free_tier_lag_days: int = Field(ge=0, le=365)


@router.post("/free-tier-lag-days")
async def set_free_tier_lag_days(body: FreeTierLagDaysUpdate):
    await set_setting_int(FREE_TIER_LAG_DAYS_KEY, body.free_tier_lag_days)
    return {"free_tier_lag_days": body.free_tier_lag_days}


@router.post("/paper-trading/enable")
async def enable_paper_trading():
    """Starts accepting paper order submissions and the two scheduler
    polling/sync jobs on their next tick — no real money is ever involved
    (Alpaca paper endpoint only), but this is the real go-live switch for
    the feature."""
    await set_setting_bool(PAPER_TRADING_ENABLED_KEY, True)
    return {"paper_trading_enabled": True}


@router.post("/paper-trading/disable")
async def disable_paper_trading():
    await set_setting_bool(PAPER_TRADING_ENABLED_KEY, False)
    return {"paper_trading_enabled": False}


@router.post("/paper-trading-kill-switch/enable")
async def enable_paper_trading_kill_switch():
    """TRD-35: blocks every new paper order submission synchronously, on
    the very next request — not delayed to a scheduler tick."""
    await set_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, True)
    return {"paper_trading_kill_switch": True}


@router.post("/paper-trading-kill-switch/disable")
async def disable_paper_trading_kill_switch():
    await set_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, False)
    return {"paper_trading_kill_switch": False}


class PaperTradingMaxOrderValueUpdate(BaseModel):
    max_order_value: float = Field(gt=0, le=10_000_000)


@router.post("/paper-trading/max-order-value")
async def set_paper_trading_max_order_value(body: PaperTradingMaxOrderValueUpdate):
    await set_setting_float(PAPER_TRADING_MAX_ORDER_VALUE_KEY, body.max_order_value)
    return {"paper_trading_max_order_value": body.max_order_value}


class PaperTradingMaxPortfolioPctUpdate(BaseModel):
    max_portfolio_pct: float = Field(gt=0, le=100)


@router.post("/paper-trading/max-portfolio-pct")
async def set_paper_trading_max_portfolio_pct(body: PaperTradingMaxPortfolioPctUpdate):
    await set_setting_float(PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY, body.max_portfolio_pct)
    return {"paper_trading_max_portfolio_pct": body.max_portfolio_pct}


class PaperTradingMaxOrdersPerDayUpdate(BaseModel):
    max_orders_per_day: int = Field(gt=0, le=10_000)


@router.post("/paper-trading/max-orders-per-day")
async def set_paper_trading_max_orders_per_day(body: PaperTradingMaxOrdersPerDayUpdate):
    await set_setting_int(PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY, body.max_orders_per_day)
    return {"paper_trading_max_orders_per_day": body.max_orders_per_day}


class PaperTradingPriceCollarPctUpdate(BaseModel):
    price_collar_pct: float = Field(gt=0, le=1000)


@router.post("/paper-trading/price-collar-pct")
async def set_paper_trading_price_collar_pct(body: PaperTradingPriceCollarPctUpdate):
    await set_setting_float(PAPER_TRADING_PRICE_COLLAR_PCT_KEY, body.price_collar_pct)
    return {"paper_trading_price_collar_pct": body.price_collar_pct}


class PaperTradingRestrictedSymbolsUpdate(BaseModel):
    restricted_symbols: str = ""


@router.post("/paper-trading/restricted-symbols")
async def set_paper_trading_restricted_symbols(body: PaperTradingRestrictedSymbolsUpdate):
    """Comma-separated tickers, e.g. 'GME,AMC'. Applies to the very next
    order submission — no restart needed."""
    await set_setting_str(PAPER_TRADING_RESTRICTED_SYMBOLS_KEY, body.restricted_symbols)
    return {"paper_trading_restricted_symbols": body.restricted_symbols}


@router.post("/stock-score-compute/enable")
async def enable_stock_score_compute():
    await set_setting_bool(STOCK_SCORE_COMPUTE_ENABLED_KEY, True)
    return {"stock_score_compute_enabled": True}


@router.post("/stock-score-compute/disable")
async def disable_stock_score_compute():
    await set_setting_bool(STOCK_SCORE_COMPUTE_ENABLED_KEY, False)
    return {"stock_score_compute_enabled": False}


class PriceDataProviderUpdate(BaseModel):
    provider: str


@router.post("/price-data-provider")
async def set_price_data_provider(body: PriceDataProviderUpdate):
    """Switches which live-quote source get_latest_price/
    get_extended_hours_price use, effective on the very next price
    request — no restart needed (see services/price_provider.py)."""
    if body.provider not in PRICE_PROVIDERS:
        raise HTTPException(422, f"provider must be one of {PRICE_PROVIDERS}")
    await set_setting_str(PRICE_DATA_PROVIDER_KEY, body.provider)
    set_price_provider(body.provider)
    return {"price_data_provider": body.provider}
