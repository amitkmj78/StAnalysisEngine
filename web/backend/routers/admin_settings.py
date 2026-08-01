from fastapi import APIRouter, Depends

from web.backend.admin import require_admin
from web.backend.app_settings import VERIFY_PREDICTIONS_ENABLED_KEY, get_setting_bool, set_setting_bool

router = APIRouter(
    prefix="/api/v1/admin/settings",
    tags=["admin-settings"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
async def get_settings():
    return {
        "verify_predictions_enabled": await get_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, default=True),
    }


@router.post("/verify-predictions/enable")
async def enable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, True)
    return {"verify_predictions_enabled": True}


@router.post("/verify-predictions/disable")
async def disable_verify_predictions():
    await set_setting_bool(VERIFY_PREDICTIONS_ENABLED_KEY, False)
    return {"verify_predictions_enabled": False}
