from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.monthly_investing_service import (
    get_best_monthly_pick,
    project_future_value,
    simulate_monthly_plan,
)
from services.stock_finder_service import STOCK_UNIVERSES
from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/monthly-plan",
    tags=["monthly-plan"],
    dependencies=[Depends(verify_bearer_token)],
)

ASSET_TYPES = {"Fund", "Stock"}
STOCK_GOALS = {"Short Term", "Long Term"}
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]


@router.get("/options")
async def options():
    return {
        "fund_goals": list(FUND_GOAL_WEIGHTS.keys()),
        "fund_categories": FUND_CATEGORIES,
        "stock_goals": list(STOCK_GOALS),
        "stock_universes": list(STOCK_UNIVERSES.keys()),
    }


@router.get("/summary")
@limiter.limit("10/minute")
async def summary(
    request: Request,
    asset_type: str = Query(...),
    goal: str = Query(...),
    selection: str = Query(...),
    monthly_amount: float = Query(1000, ge=100, le=10000),
    years: int = Query(5, ge=1, le=15),
):
    await enforce_daily_quota(request, "monthly-plan/summary")

    if asset_type not in ASSET_TYPES:
        raise HTTPException(422, f"asset_type must be one of {sorted(ASSET_TYPES)}")
    if asset_type == "Fund":
        if goal not in FUND_GOAL_WEIGHTS:
            raise HTTPException(422, f"goal must be one of {sorted(FUND_GOAL_WEIGHTS.keys())}")
        if selection not in FUND_CATEGORIES:
            raise HTTPException(422, f"selection must be one of {FUND_CATEGORIES}")
    else:
        if goal not in STOCK_GOALS:
            raise HTTPException(422, f"goal must be one of {sorted(STOCK_GOALS)}")
        if selection not in STOCK_UNIVERSES:
            raise HTTPException(422, f"selection must be one of {sorted(STOCK_UNIVERSES.keys())}")

    recommendation = await run_in_threadpool(get_best_monthly_pick, asset_type, goal, selection)
    if recommendation is None:
        return {"recommendation": None, "history": None, "summary": None, "projected_value": None}

    history_df, plan_summary = await run_in_threadpool(
        simulate_monthly_plan, recommendation.ticker, float(monthly_amount), years
    )
    projected_value = project_future_value(float(monthly_amount), years, recommendation.expected_return_pct)

    rec_out = {
        "ticker": recommendation.ticker,
        "name": recommendation.name,
        "score": recommendation.score,
        "asset_type": recommendation.asset_type,
        "expected_return_pct": recommendation.expected_return_pct,
    }

    if not plan_summary:
        return {"recommendation": rec_out, "history": None, "summary": None, "projected_value": projected_value}

    history = {
        "dates": history_df["Date"].dt.strftime("%Y-%m").tolist(),
        "contribution": history_df["Monthly Contribution"].round(2).tolist(),
        "price": history_df["Price"].round(2).tolist(),
        "shares_bought": history_df["Shares Bought"].round(4).tolist(),
        "total_invested": history_df["Total Invested"].round(2).tolist(),
        "portfolio_value": history_df["Portfolio Value"].round(2).tolist(),
    }

    return {
        "recommendation": rec_out,
        "history": history,
        "summary": plan_summary,
        "projected_value": projected_value,
    }
