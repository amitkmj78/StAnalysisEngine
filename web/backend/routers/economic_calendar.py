from fastapi import APIRouter, Depends

from services.economic_calendar_service import DISCLOSURE, get_upcoming_economic_events
from web.backend.app_settings import ECONOMIC_CALENDAR_ENABLED_KEY, get_setting_bool
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn

router = APIRouter(prefix="/api/v1/market", tags=["economic-calendar"])


@router.get("/economic-calendar", dependencies=[Depends(verify_bearer_token)])
async def get_economic_calendar():
    """EconomicCalendarBanner.tsx's backing endpoint. Returns {"available":
    false, "reason": ...} rather than erroring when the feature is off --
    same convention as GET /api/v1/market/regime -- never when there's
    simply nothing in the window; `upcoming` is just empty then, which the
    banner also treats as "render nothing."""
    if not await get_setting_bool(ECONOMIC_CALENDAR_ENABLED_KEY, default=False):
        return {"available": False, "reason": "economic_calendar_enabled is off"}

    async with service_conn() as conn:
        rows = await conn.fetch("SELECT event_type, event_date, title, note FROM economic_event_overrides")
    overrides = [dict(r) for r in rows]

    return {
        "available": True,
        "upcoming": get_upcoming_economic_events(overrides),
        "disclosure": DISCLOSURE,
    }
