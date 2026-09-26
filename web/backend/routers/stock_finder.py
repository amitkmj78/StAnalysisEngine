import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.analyst_rating_service import get_analyst_rating_summary
from services.stock_finder_service import (
    SECTOR_WEIGHTING_MODES,
    SP500_UNIVERSE_NAME,
    STOCK_UNIVERSES,
    build_diversified_basket,
    generate_diversified_basket,
    get_universe_sector_preview,
    rank_stocks,
    score_stock_ticker,
)
from services.subscriber_events_service import log_event

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

# One-line descriptions for the Build-a-Basket universe picker (DI-01).
# "All" and S&P 500 are resolved live (Wikipedia fetch / union of every
# universe below); the other two are small hardcoded samples, not the
# real ETF's full holdings -- said plainly here rather than implied.
UNIVERSE_DESCRIPTIONS: dict[str, str] = {
    "All": "Every ticker across every universe below, deduplicated.",
    SP500_UNIVERSE_NAME: "The full S&P 500, refreshed daily from a live constituent list.",
    "US - Mega Cap (SPY sample)": "A small, hardcoded sample of the largest S&P 500 names — not the full SPY holdings list.",
    "US - Tech Growth (QQQ sample)": "A small, hardcoded sample of large tech/growth names — not the full QQQ holdings list.",
}

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


@router.get("/universes/detail")
async def universes_detail():
    """Per-universe description, live stock count, sector breakdown, and
    as-of date (DI-01) for the Build-a-Basket page's picker -- a new,
    additive endpoint; does NOT change /universes's plain string-list
    contract, which the plain Stock Screener page and GoalPlan.tsx still
    rely on."""
    out = []
    for key in STOCK_UNIVERSES:
        preview = await run_in_threadpool(get_universe_sector_preview, key)
        out.append({"key": key, "description": UNIVERSE_DESCRIPTIONS.get(key, ""), **preview})
    return {"universes": out}


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
    max_stocks: Optional[int] = Query(None, ge=1, le=100),
):
    """DEPRECATED — kept only for backward compatibility with any existing
    integration. See POST /diversified-basket/preview for the current
    "Build a Diversified Basket" page, which adds exclusion-reason
    tracking, sub-industry capping, whole-share/fractional sizing, a
    sector summary, a risk preview, and a concentration warning that this
    endpoint has none of.

    A custom, sector-diversified basket of individual stocks — the
    picks_per_sector highest-Score tickers from each sector in the
    universe, optionally capped at max_stocks total (round-robin across
    sectors — see build_diversified_basket). Same cost profile as /rank
    (it calls it under the hood), so same tight quota."""
    await enforce_daily_quota(request, "stock-finder/diversified-basket")
    _validate_goal(goal)
    if universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    df = await run_in_threadpool(build_diversified_basket, goal, universe, picks_per_sector, max_stocks)
    return {"results": records_safe(df)}


class DiversifiedBasketRequest(BaseModel):
    goal: str
    universe: str
    picks_per_sector: int = Field(2, ge=1, le=10)
    max_stocks: Optional[int] = Field(None, ge=1, le=100)
    total_amount: float = Field(10_000, ge=100, le=10_000_000)
    fractional_shares: bool = False
    sector_weighting: str = "equal_dollar"
    excluded_tickers: list[str] = []


@router.post("/diversified-basket/preview")
@limiter.limit("10/minute")
async def diversified_basket_preview(request: Request, body: DiversifiedBasketRequest):
    """Full "Build a Diversified Basket" generation (DI-01 through DI-09,
    plus sub-industry capping, sector-weighting choice, and a risk
    preview) — see services.stock_finder_service.generate_diversified_basket
    for the algorithm. Same cost profile as /rank, so the same tight quota."""
    await enforce_daily_quota(request, "stock-finder/diversified-basket/preview")
    _validate_goal(body.goal)
    if body.universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")
    if body.sector_weighting not in SECTOR_WEIGHTING_MODES:
        raise HTTPException(422, f"sector_weighting must be one of {SECTOR_WEIGHTING_MODES}")

    result = await run_in_threadpool(
        generate_diversified_basket,
        body.goal, body.universe, body.picks_per_sector, body.max_stocks, body.total_amount,
        body.fractional_shares, body.sector_weighting, body.excluded_tickers,
    )

    # Every generation logged (inputs, as-of date, resulting tickers) for
    # support/backtesting -- the non-functional requirement in the spec.
    await log_event(
        request.state.user["id"],
        "diversified_basket_generated",
        resource=body.universe,
        metadata={
            "goal": body.goal, "universe": body.universe, "picks_per_sector": body.picks_per_sector,
            "max_stocks": body.max_stocks, "total_amount": body.total_amount,
            "sector_weighting": body.sector_weighting, "fractional_shares": body.fractional_shares,
            "as_of_date": result["as_of_date"], "tickers": [h["Ticker"] for h in result["holdings"]],
        },
    )
    return result


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
