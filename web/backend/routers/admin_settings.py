from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from web.backend.admin import require_admin
from web.backend.app_settings import (
    DAILY_QUOTA_DEFAULT,
    DAILY_QUOTA_KEY,
    DB_BACKUP_ENABLED_KEY,
    FREE_TIER_LAG_DAYS_DEFAULT,
    FREE_TIER_LAG_DAYS_KEY,
    HORIZON1_SUBSCRIPTIONS_ENABLED_KEY,
    PASSWORD_POLICY_ENABLED_KEY,
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
    get_setting_int,
    set_setting_bool,
    set_setting_float,
    set_setting_int,
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
    }


@router.post("/verify-predictions/enable")
async def enable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, True)
    return {"verify_predictions_enabled": True}


@router.post("/verify-predictions/disable")
async def disable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, False)
    return {"verify_predictions_enabled": False}


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
    """Starts real per-drop Tavily + LLM calls and user-visible in-app
    notifications on the next scheduler tick — not just a preview toggle."""
    await set_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, True)
    return {"portfolio_drop_alerts_enabled": True}


@router.post("/portfolio-drop-alerts/disable")
async def disable_portfolio_drop_alerts():
    await set_setting_bool(PORTFOLIO_DROP_ALERTS_ENABLED_KEY, False)
    return {"portfolio_drop_alerts_enabled": False}


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
