from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.index_fund_service import GOAL_WEIGHTS, rank_index_funds, score_fund_ticker

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/index-fund",
    tags=["index-fund"],
    dependencies=[Depends(verify_bearer_token)],
)

FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]


def _validate_goal(goal: str) -> None:
    if goal not in GOAL_WEIGHTS:
        raise HTTPException(422, f"goal must be one of {sorted(GOAL_WEIGHTS.keys())}")


@router.get("/goals")
async def goals():
    return {"goals": list(GOAL_WEIGHTS.keys())}


@router.get("/categories")
async def categories():
    return {"categories": FUND_CATEGORIES}


@router.get("/rank")
@limiter.limit("10/minute")
async def rank(request: Request, goal: str = Query(...), category: str = Query("All")):
    await enforce_daily_quota(request, "index-fund/rank")
    _validate_goal(goal)
    if category not in FUND_CATEGORIES:
        raise HTTPException(422, f"category must be one of {FUND_CATEGORIES}")

    df = await run_in_threadpool(rank_index_funds, goal, category)
    return {"results": records_safe(df)}


@router.get("/score")
@limiter.limit("20/minute")
async def score(request: Request, goal: str = Query(...), ticker: str = Query(..., min_length=1)):
    await enforce_daily_quota(request, "index-fund/score")
    _validate_goal(goal)
    ticker = ticker.strip().upper()

    df = await run_in_threadpool(score_fund_ticker, goal, ticker)
    records = records_safe(df)
    return {"result": records[0] if records else None}
