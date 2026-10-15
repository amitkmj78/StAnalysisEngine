"""Admin CRUD for economic_event_overrides -- the only source for FOMC
meeting dates (see services/economic_calendar_service.py's module
docstring for why FRED isn't used for those), and a manual corrective for
any other event type FRED misses."""

from datetime import date

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.economic_calendar_service import EVENT_TYPE_TITLES
from web.backend.admin import require_admin
from web.backend.db import service_conn

router = APIRouter(
    prefix="/api/v1/admin/economic-calendar",
    tags=["admin-economic-calendar"],
    dependencies=[Depends(require_admin)],
)


def _record_to_dict(record) -> dict:
    row = {k: record[k] for k in record.keys()}
    row["event_date"] = row["event_date"].isoformat()
    row["id"] = str(row["id"])
    return row


class EconomicEventOverrideRequest(BaseModel):
    event_type: str
    event_date: date
    title: str = Field(min_length=1, max_length=200)
    note: str | None = None


@router.get("/overrides")
async def list_economic_event_overrides():
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM economic_event_overrides ORDER BY event_date")
    return {"overrides": [_record_to_dict(r) for r in rows]}


@router.post("/overrides")
async def create_economic_event_override(body: EconomicEventOverrideRequest):
    if body.event_type not in EVENT_TYPE_TITLES:
        raise HTTPException(422, f"event_type must be one of {sorted(EVENT_TYPE_TITLES)}.")
    async with service_conn() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO economic_event_overrides (event_type, event_date, title, note)
                VALUES ($1, $2, $3, $4) RETURNING *
                """,
                body.event_type, body.event_date, body.title.strip(), body.note,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(409, "An override for that event type and date already exists.")
    return _record_to_dict(row)


@router.delete("/overrides/{override_id}")
async def delete_economic_event_override(override_id: str):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "DELETE FROM economic_event_overrides WHERE id = $1 RETURNING id", override_id
        )
    if row is None:
        raise HTTPException(404, "Override not found.")
    return {"ok": True}
