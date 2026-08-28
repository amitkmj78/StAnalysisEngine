import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.analyst_rating_service import get_analyst_rating_summary
from services.stock_finder_service import STOCK_UNIVERSES, build_diversified_basket, rank_stocks, score_stock_ticker

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/stock-finder",
    tags=["stock-finder"],
    dependencies=[Depends(verify_bearer_token)],
)

ALLOWED_GOALS = {"Short Term", "Long Term"}


def _validate_goal(goal: str) -> None:
    if goal not in ALLOWED_GOALS:
        raise HTTPException(422, f"goal must be one of {sorted(ALLOWED_GOALS)}")


@router.get("/universes")
async def universes():
    return {"universes": list(STOCK_UNIVERSES.keys())}


@router.get("/rank")
@limiter.limit("10/minute")
async def rank(
    request: Request,
    goal: str = Query(...),
    universe: str = Query("All"),
):
    # Tighter limit than /score: a single call fans out to ~yfinance calls
    # per ticker in the universe (see services/stock_finder_service.py).
    await enforce_daily_quota(request, "stock-finder/rank")
    _validate_goal(goal)
    if universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    df = await run_in_threadpool(rank_stocks, goal, universe)
    return {"results": records_safe(df)}


@router.get("/score")
@limiter.limit("20/minute")
async def score(
    request: Request,
    goal: str = Query(...),
    ticker: str = Query(..., min_length=1),
):
    await enforce_daily_quota(request, "stock-finder/score")
    _validate_goal(goal)
    ticker = ticker.strip().upper()

    df = await run_in_threadpool(score_stock_ticker, goal, ticker)
    records = records_safe(df)
    return {"result": records[0] if records else None}


@router.get("/diversified-basket")
@limiter.limit("10/minute")
async def diversified_basket(
    request: Request,
    goal: str = Query(...),
    universe: str = Query("All"),
    picks_per_sector: int = Query(2, ge=1, le=10),
):
    """A custom, sector-diversified basket of individual stocks — the
    picks_per_sector highest-Score tickers from each sector in the
    universe. Same cost profile as /rank (it calls it under the hood), so
    same tight quota."""
    await enforce_daily_quota(request, "stock-finder/diversified-basket")
    _validate_goal(goal)
    if universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    df = await run_in_threadpool(build_diversified_basket, goal, universe, picks_per_sector)
    return {"results": records_safe(df)}


@router.get("/analyst")
@limiter.limit("20/minute")
async def analyst(request: Request, ticker: str = Query(..., min_length=1)):
    """Real, third-party Wall Street analyst consensus + price targets for
    one ticker — see services/analyst_rating_service.py. Not part of /rank
    (one yfinance call per ticker — too slow to run for a whole scanned
    universe), fetched on demand per row instead, same pattern as the
    screener's Quant Signal column."""
    await enforce_daily_quota(request, "stock-finder/analyst")
    ticker = ticker.strip().upper()
    result = await run_in_threadpool(get_analyst_rating_summary, ticker)
    if result is None:
        raise HTTPException(404, f"No analyst coverage found for {ticker}.")
    return result


# --- Saved screens (US-04/05, docs/stock-screener-improvements-spec.md):
# a named, reloadable goal+universe+filters+columns+sort configuration, plus
# a top-10 snapshot at save time so a later reload can show what moved.

def _screen_to_dict(record) -> dict:
    row = {k: record[k] for k in record.keys()}
    for col in ("filters", "visible_columns", "sort_keys", "snapshot_top10"):
        if isinstance(row.get(col), str):
            row[col] = json.loads(row[col])
    return row


class SortKey(BaseModel):
    column: str
    direction: str


class SaveScreenRequest(BaseModel):
    name: str
    goal: str
    universe: str
    filters: dict[str, Any] = {}
    visible_columns: list[str] = []
    sort_keys: list[SortKey] = []
    snapshot_top10: list[dict[str, Any]] = []


@router.post("/screens/save")
@limiter.limit("20/minute")
async def save_screen(request: Request, body: SaveScreenRequest):
    await enforce_daily_quota(request, "stock-finder/screens/save")
    user_id = request.state.user["id"]

    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name is required")
    _validate_goal(body.goal)
    if body.universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO saved_screens (
                user_id, name, goal, universe, filters, visible_columns, sort_keys, snapshot_top10
            ) VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb)
            RETURNING *
            """,
            user_id, name, body.goal, body.universe,
            json.dumps(body.filters), json.dumps(body.visible_columns),
            json.dumps([k.model_dump() for k in body.sort_keys]), json.dumps(body.snapshot_top10),
        )
    return {"screen": _screen_to_dict(record)}


@router.get("/screens")
async def list_screens(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM saved_screens ORDER BY saved_at DESC")
    return {"screens": [_screen_to_dict(r) for r in records]}


@router.delete("/screens/{screen_id}")
async def delete_screen(request: Request, screen_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM saved_screens WHERE id = $1 RETURNING id",
            screen_id,
        )
    if row is None:
        raise HTTPException(404, "Saved screen not found.")
    return {"ok": True}
