"""
ALR-1/2: per-alert-type channel preferences (global or per-ticker) and
notification settings (quiet hours, daily digest, webhook -- ALR-3's
enable/URL live here too, added now per the Smart Alerts plan's Stage 8
decision to avoid a second migration; webhook_secret isn't settable from
this router yet, that lands with the actual sender in Stage 11).
"""

from datetime import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.notification_dispatcher import DEFAULT_PREFERENCE
from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn

router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["alert-preferences"],
    dependencies=[Depends(verify_bearer_token)],
)

# Matches services/notification_dispatcher.py's alert_type strings.
ALERT_TYPES = {"signal_change", "earnings", "cost_drop"}


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


def _parse_time(value: Optional[str]) -> Optional[time]:
    if value is None or value == "":
        return None
    try:
        return time.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"'{value}' is not a valid HH:MM time.")


class PreferenceUpsertRequest(BaseModel):
    alert_type: str
    ticker: Optional[str] = None
    enabled: bool = True
    channel_email: bool = True
    channel_inapp: bool = True


@router.get("/preferences")
async def list_preferences(request: Request):
    """Every saved preference row for this user (global + per-ticker).
    A ticker/alert_type with no row here uses notification_dispatcher's
    default (enabled, email+in-app on) -- this endpoint only returns
    explicit overrides, not a synthesized full matrix."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT * FROM user_alert_preferences ORDER BY ticker NULLS FIRST, alert_type"
        )
    return {"default": DEFAULT_PREFERENCE, "overrides": [_record_to_dict(r) for r in rows]}


@router.put("/preferences")
async def upsert_preference(request: Request, body: PreferenceUpsertRequest):
    """Sets the global (no ticker) or per-ticker preference for one
    alert_type -- a per-ticker row overrides the global row for that
    ticker only, matching ALR-1's "user chooses alert types per stock or
    globally" acceptance criterion."""
    user_id = request.state.user["id"]
    if body.alert_type not in ALERT_TYPES:
        raise HTTPException(422, f"alert_type must be one of {sorted(ALERT_TYPES)}")
    ticker = body.ticker.strip().upper() if body.ticker else None

    async with user_conn(user_id) as conn:
        if ticker is None:
            record = await conn.fetchrow(
                """
                INSERT INTO user_alert_preferences (user_id, ticker, alert_type, enabled, channel_email, channel_inapp)
                VALUES ($1::uuid, NULL, $2, $3, $4, $5)
                ON CONFLICT (user_id, alert_type) WHERE ticker IS NULL
                DO UPDATE SET enabled = $3, channel_email = $4, channel_inapp = $5, updated_at = now()
                RETURNING *
                """,
                user_id, body.alert_type, body.enabled, body.channel_email, body.channel_inapp,
            )
        else:
            record = await conn.fetchrow(
                """
                INSERT INTO user_alert_preferences (user_id, ticker, alert_type, enabled, channel_email, channel_inapp)
                VALUES ($1::uuid, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, ticker, alert_type) WHERE ticker IS NOT NULL
                DO UPDATE SET enabled = $4, channel_email = $5, channel_inapp = $6, updated_at = now()
                RETURNING *
                """,
                user_id, ticker, body.alert_type, body.enabled, body.channel_email, body.channel_inapp,
            )
    return _record_to_dict(record)


@router.delete("/preferences/{preference_id}")
async def delete_preference(request: Request, preference_id: int):
    """Removes an override, reverting that ticker/alert_type (or the
    global default for that alert_type) back to notification_dispatcher's
    built-in default."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("DELETE FROM user_alert_preferences WHERE id = $1 RETURNING id", preference_id)
    if row is None:
        raise HTTPException(404, "Preference not found.")
    return {"ok": True}


class SettingsUpsertRequest(BaseModel):
    quiet_hours_start: Optional[str] = None  # "HH:MM", null clears it
    quiet_hours_end: Optional[str] = None
    digest_enabled: bool = False
    digest_time: str = "08:00"
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None


def _settings_response(record) -> dict:
    result = _record_to_dict(record)
    result["has_webhook_secret"] = bool(result.pop("webhook_secret", None))
    for key in ("quiet_hours_start", "quiet_hours_end", "digest_time"):
        if result.get(key) is not None:
            result[key] = result[key].isoformat(timespec="minutes")
    return result


_SETTINGS_DEFAULT = {
    "quiet_hours_start": None, "quiet_hours_end": None, "digest_enabled": False,
    "digest_time": "08:00", "webhook_enabled": False, "webhook_url": None,
    "has_webhook_secret": False,
}


@router.get("/settings")
async def get_settings(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("SELECT * FROM user_notification_settings WHERE user_id = $1::uuid", user_id)
    return _settings_response(row) if row is not None else dict(_SETTINGS_DEFAULT)


@router.put("/settings")
async def upsert_settings(request: Request, body: SettingsUpsertRequest):
    user_id = request.state.user["id"]
    quiet_start = _parse_time(body.quiet_hours_start)
    quiet_end = _parse_time(body.quiet_hours_end)
    digest_time = _parse_time(body.digest_time) or time(8, 0)

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO user_notification_settings (
                user_id, quiet_hours_start, quiet_hours_end, digest_enabled, digest_time,
                webhook_enabled, webhook_url
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id) DO UPDATE SET
                quiet_hours_start = $2, quiet_hours_end = $3, digest_enabled = $4, digest_time = $5,
                webhook_enabled = $6, webhook_url = $7, updated_at = now()
            RETURNING *
            """,
            user_id, quiet_start, quiet_end, body.digest_enabled, digest_time,
            body.webhook_enabled, body.webhook_url,
        )
    return _settings_response(record)
