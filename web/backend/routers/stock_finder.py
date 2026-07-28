from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.stock_finder_service import STOCK_UNIVERSES, rank_stocks, score_stock_ticker

from web.backend.auth import verify_bearer_token
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
