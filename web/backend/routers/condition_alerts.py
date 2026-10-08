import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.condition_alert_service import (
    CATEGORY_FIELDS,
    COMBINATORS,
    MAX_CONDITIONS,
    NUMERIC_FIELDS,
    parse_conditions,
)
from services.strategy_engine import NUMERIC_OPS, CATEGORY_OPS

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/condition-alerts",
    tags=["condition-alerts"],
    dependencies=[Depends(verify_bearer_token)],
)


class ConditionIn(BaseModel):
    field: str
    op: str
    value: object


class ConditionAlertCreateRequest(BaseModel):
    ticker: str
    conditions: list[ConditionIn]
    combinator: str


def _record_to_dict(record) -> dict:
    row = {k: record[k] for k in record.keys()}
    if isinstance(row.get("conditions"), str):
        row["conditions"] = json.loads(row["conditions"])
    return row


@router.get("/fields")
async def list_fields():
    """ALX-1: the field/op vocabulary a condition can reference, for the
    frontend's condition-builder dropdowns -- one source of truth, same
    one parse_condition validates against server-side."""
    return {
        "numeric_fields": NUMERIC_FIELDS,
        "numeric_ops": sorted(NUMERIC_OPS),
        "category_fields": {k: list(v) for k, v in CATEGORY_FIELDS.items()},
        "category_ops": sorted(CATEGORY_OPS),
        "combinators": sorted(COMBINATORS),
        "max_conditions": MAX_CONDITIONS,
    }


@router.get("")
async def list_alerts(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch("SELECT * FROM condition_alerts ORDER BY created_at DESC")
    return [_record_to_dict(r) for r in rows]


@router.post("")
@limiter.limit("20/minute")
async def create_alert(request: Request, body: ConditionAlertCreateRequest):
    await enforce_daily_quota(request, "condition-alerts/create")
    user_id = request.state.user["id"]

    if body.combinator not in COMBINATORS:
        raise HTTPException(422, f"combinator must be one of {sorted(COMBINATORS)}")
    try:
        parse_conditions([c.model_dump() for c in body.conditions])
    except ValueError as e:
        raise HTTPException(422, str(e))

    ticker = body.ticker.strip().upper()
    if not ticker:
        raise HTTPException(422, "ticker is required")

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO condition_alerts (user_id, ticker, conditions, combinator)
            VALUES ($1::uuid, $2, $3::jsonb, $4)
            RETURNING *
            """,
            user_id, ticker, json.dumps([c.model_dump() for c in body.conditions]), body.combinator,
        )
    return _record_to_dict(record)


@router.delete("/{alert_id}")
async def delete_alert(request: Request, alert_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("DELETE FROM condition_alerts WHERE id = $1 RETURNING id", alert_id)
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(request: Request, alert_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE condition_alerts SET seen_at = now() WHERE id = $1 RETURNING id",
            alert_id,
        )
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}
