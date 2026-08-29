from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.fund_comparison_service import price_near_date
from services.index_fund_service import GOAL_WEIGHTS, rank_index_funds, score_fund_ticker

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/index-fund",
    tags=["index-fund"],
    dependencies=[Depends(verify_bearer_token)],
)

FUND_CATEGORIES = [
    "All",
    "US Large Blend",
    "US Total Market",
    "US Large Growth",
    "US Large Value",
    "US Mid Cap",
    "US Small Cap",
    "International Developed",
    "International Total",
    "Emerging Markets",
    "Bond — Total Market",
    "Bond — Short-Term",
    "Bond — Long-Term/Treasury",
    "Bond — Corporate",
    "Bond — High Yield",
    "Bond — TIPS",
    "Dividend/Income",
    "Real Estate",
    "Sector",
]


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


@router.get("/return-since")
@limiter.limit("20/minute")
async def return_since(request: Request, ticker: str = Query(..., min_length=1), since: date = Query(...)):
    """
    Real, point-in-time return for one ticker from `since` to now — "what
    if you'd put this money in this fund instead, starting the same day,"
    not a fixed 30d/1Y/3Y window that may not match how long the caller has
    actually been invested. `since` can be decades back (e.g. a fund's
    inception date) — price_near_date is asked for "max" history, not the
    2y default, to cover that.
    """
    await enforce_daily_quota(request, "index-fund/return-since")
    ticker = ticker.strip().upper()
    since_dt = datetime.combine(since, datetime.min.time())

    price_then = await run_in_threadpool(price_near_date, ticker, since_dt, "max")
    price_now = await run_in_threadpool(get_latest_price, ticker)
    if price_then is None or price_now is None:
        raise HTTPException(404, f"No price history found for {ticker}.")

    return {
        "ticker": ticker,
        "since": str(since),
        "days": (date.today() - since).days,
        "price_then": round(price_then, 2),
        "price_now": round(price_now, 2),
        "return_pct": round((price_now - price_then) / price_then * 100, 2) if price_then else None,
    }


@router.get("/score")
@limiter.limit("20/minute")
async def score(request: Request, goal: str = Query(...), ticker: str = Query(..., min_length=1)):
    await enforce_daily_quota(request, "index-fund/score")
    _validate_goal(goal)
    ticker = ticker.strip().upper()

    df = await run_in_threadpool(score_fund_ticker, goal, ticker)
    records = records_safe(df)
    return {"result": records[0] if records else None}
