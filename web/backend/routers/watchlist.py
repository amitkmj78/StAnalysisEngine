from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.alert_engine_service import CONDITION_TYPES

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/watchlist",
    tags=["watchlist"],
    dependencies=[Depends(verify_bearer_token)],
)


class AlertCreateRequest(BaseModel):
    ticker: str
    condition_type: str
    threshold: float


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


@router.get("")
async def list_alerts(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch("SELECT * FROM watchlist_alerts ORDER BY created_at DESC")
    return [_record_to_dict(r) for r in rows]


@router.post("")
@limiter.limit("20/minute")
async def create_alert(request: Request, body: AlertCreateRequest):
    await enforce_daily_quota(request, "watchlist/create")
    user_id = request.state.user["id"]

    if body.condition_type not in CONDITION_TYPES:
        raise HTTPException(422, f"condition_type must be one of {sorted(CONDITION_TYPES)}")
    if body.threshold <= 0:
        raise HTTPException(422, "threshold must be positive")

    ticker = body.ticker.strip().upper()
    if not ticker:
        raise HTTPException(422, "ticker is required")

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO watchlist_alerts (user_id, ticker, condition_type, threshold)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING *
            """,
            user_id, ticker, body.condition_type, body.threshold,
        )
    return _record_to_dict(record)


@router.delete("/{alert_id}")
async def delete_alert(request: Request, alert_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("DELETE FROM watchlist_alerts WHERE id = $1 RETURNING id", alert_id)
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(request: Request, alert_id: int):
    """Clears the "new" state on a triggered alert without deleting its history."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE watchlist_alerts SET seen_at = now() WHERE id = $1 RETURNING id",
            alert_id,
        )
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}
