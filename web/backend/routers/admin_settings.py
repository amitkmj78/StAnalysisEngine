from fastapi import APIRouter, Depends

from web.backend.admin import require_admin
from web.backend.app_settings import (
    PASSWORD_POLICY_ENABLED_KEY,
    PUBLISH_SIGNALS_ENABLED_KEY,
    VERIFY_PREDICTIONS_ENABLED_KEY,
    get_setting_bool,
    set_setting_bool,
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
