"""CHT-7: saved chart layouts per user. A layout is stored as JSON and returned unchanged, so it reloads identically."""

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/chart-layouts", tags=["chart-layouts"], dependencies=[Depends(verify_bearer_token)])

MAX_TICKERS = 4


class ChartLayout(BaseModel):
    tickers: list[str] = Field(default_factory=list, max_length=MAX_TICKERS)
    range: Literal["1M", "6M", "1Y", "5Y"] = "1Y"
    chart_type: Literal["line", "candles"] = "line"
    log_scale: bool = False
    linked_crosshair: bool = True


class SaveLayout(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    layout: ChartLayout


def _as_dict(value) -> dict:
    return json.loads(value) if isinstance(value, str) else dict(value)


@router.get("")
@limiter.limit("60/minute")
async def list_layouts(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch("SELECT id, name, layout, updated_at FROM chart_layouts ORDER BY updated_at DESC LIMIT 50")
    return {
        "layouts": [
            {"id": r["id"], "name": r["name"], "layout": _as_dict(r["layout"]), "updated_at": r["updated_at"].isoformat()}
            for r in rows
        ]
    }


@router.post("")
@limiter.limit("30/minute")
async def save_layout(request: Request, body: SaveLayout):
    """Saves under a name. Saving the same name again replaces that layout."""
    user_id = request.state.user["id"]
    tickers = [t.strip().upper() for t in body.layout.tickers if t.strip()]
    if len(set(tickers)) != len(tickers):
        raise HTTPException(422, "Each chart needs a different stock.")
    layout = body.layout.model_copy(update={"tickers": tickers}).model_dump()
    async with user_conn(user_id) as conn:
        new_id = await conn.fetchval(
            """
            INSERT INTO chart_layouts (user_id, name, layout)
            VALUES ($1::uuid, $2, $3::jsonb)
            ON CONFLICT (user_id, name) DO UPDATE SET layout = EXCLUDED.layout, updated_at = now()
            RETURNING id
            """,
            user_id, body.name.strip(), json.dumps(layout),
        )
    return {"id": new_id, "layout": layout}


@router.delete("/{layout_id}")
@limiter.limit("30/minute")
async def delete_layout(request: Request, layout_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        status = await conn.execute("DELETE FROM chart_layouts WHERE id = $1", layout_id)
    if status.endswith(" 0"):
        raise HTTPException(404, "Saved layout not found.")
    return {"ok": True}
