"""CHT-5: chart drawings per user and ticker. Each drawing is a list of points on the price chart, stored as JSON
so it can be redrawn exactly. A point is a date or timestamp label (x) and a price (y)."""

import json
import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/chart-drawings", tags=["chart-drawings"], dependencies=[Depends(verify_bearer_token)])

TICKER_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")
POINTS_PER_KIND = {"trend": 2, "horizontal": 1, "rectangle": 2, "fibonacci": 2, "text": 1}
MAX_DRAWINGS_PER_TICKER = 100


class DrawingPoint(BaseModel):
    x: str = Field(min_length=1, max_length=40)
    y: float


class NewDrawing(BaseModel):
    kind: Literal["trend", "horizontal", "rectangle", "fibonacci", "text"]
    points: list[DrawingPoint] = Field(min_length=1, max_length=2)
    text: str | None = Field(default=None, max_length=200)


def _ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise HTTPException(422, "Not a valid ticker.")
    return ticker


def _as_row(r) -> dict:
    # asyncpg returns jsonb as text unless a codec is set, so decode it here to a list.
    points = json.loads(r["points"]) if isinstance(r["points"], str) else r["points"]
    return {
        "id": r["id"],
        "ticker": r["ticker"],
        "kind": r["kind"],
        "points": points,
        "text": r["text"],
    }


@router.get("/{ticker}")
@limiter.limit("60/minute")
async def list_drawings(request: Request, ticker: str):
    user_id = request.state.user["id"]
    ticker = _ticker(ticker)
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT id, ticker, kind, points, text FROM chart_drawings WHERE ticker = $1 ORDER BY id",
            ticker,
        )
    return {"ticker": ticker, "drawings": [_as_row(r) for r in rows]}


@router.post("/{ticker}")
@limiter.limit("60/minute")
async def create_drawing(request: Request, ticker: str, body: NewDrawing):
    user_id = request.state.user["id"]
    ticker = _ticker(ticker)
    if len(body.points) != POINTS_PER_KIND[body.kind]:
        raise HTTPException(422, f"A {body.kind} drawing needs {POINTS_PER_KIND[body.kind]} point(s).")
    if body.kind == "text" and not (body.text or "").strip():
        raise HTTPException(422, "A text note needs some text.")
    if body.kind != "text" and body.text is not None:
        raise HTTPException(422, "Only a text note carries text.")

    points = [p.model_dump() for p in body.points]
    async with user_conn(user_id) as conn:
        count = await conn.fetchval("SELECT count(*) FROM chart_drawings WHERE ticker = $1", ticker)
        if count >= MAX_DRAWINGS_PER_TICKER:
            raise HTTPException(409, f"You can keep up to {MAX_DRAWINGS_PER_TICKER} drawings per stock. Delete one first.")
        row = await conn.fetchrow(
            """
            INSERT INTO chart_drawings (user_id, ticker, kind, points, text)
            VALUES ($1::uuid, $2, $3, $4::jsonb, $5)
            RETURNING id, ticker, kind, points, text
            """,
            user_id, ticker, body.kind, json.dumps(points), (body.text or "").strip() or None,
        )
    return _as_row(row)


@router.delete("/item/{drawing_id}")
@limiter.limit("60/minute")
async def delete_drawing(request: Request, drawing_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        deleted = await conn.fetchval("DELETE FROM chart_drawings WHERE id = $1 RETURNING id", drawing_id)
    if deleted is None:
        raise HTTPException(404, "Drawing not found.")
    return {"deleted": True}


@router.delete("/{ticker}")
@limiter.limit("20/minute")
async def clear_drawings(request: Request, ticker: str):
    user_id = request.state.user["id"]
    ticker = _ticker(ticker)
    async with user_conn(user_id) as conn:
        result = await conn.execute("DELETE FROM chart_drawings WHERE ticker = $1", ticker)
    return {"ticker": ticker, "deleted": int(result.split()[-1])}
