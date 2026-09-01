from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.monthly_investing_service import (
    get_best_monthly_pick,
    project_future_value,
    simulate_monthly_plan,
)
from services.stock_finder_service import STOCK_UNIVERSES
from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/monthly-plan",
    tags=["monthly-plan"],
    dependencies=[Depends(verify_bearer_token)],
)

ASSET_TYPES = {"Fund", "Stock"}
STOCK_GOALS = {"Short Term", "Long Term"}
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


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


class SaveMonthlyPlanRequest(BaseModel):
    name: str = "Monthly Plan"
    monthly_amount: float
    years: int
    fund_goal: str
    fund_category: str
    stock_goal: str
    stock_universe: str


@router.post("/saved")
async def save_monthly_plan(request: Request, body: SaveMonthlyPlanRequest):
    """Saves the inputs to the Monthly Investing Plan form — not a frozen
    result. Re-loading a saved plan (GET /monthly-plan/summary with these
    same params, for both the fund and stock side) always reflects
    today's rankings/prices, same as re-typing the same values in and
    hitting Build Plan again."""
    if body.monthly_amount < 100 or body.monthly_amount > 10000:
        raise HTTPException(422, "monthly_amount must be between 100 and 10,000.")
    if body.years < 1 or body.years > 15:
        raise HTTPException(422, "years must be between 1 and 15.")
    if body.fund_goal not in FUND_GOAL_WEIGHTS:
        raise HTTPException(422, f"fund_goal must be one of {sorted(FUND_GOAL_WEIGHTS.keys())}")
    if body.fund_category not in FUND_CATEGORIES:
        raise HTTPException(422, f"fund_category must be one of {FUND_CATEGORIES}")
    if body.stock_goal not in STOCK_GOALS:
        raise HTTPException(422, f"stock_goal must be one of {sorted(STOCK_GOALS)}")
    if body.stock_universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"stock_universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO saved_monthly_plans
                (user_id, name, monthly_amount, years, fund_goal, fund_category, stock_goal, stock_universe)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            user_id, body.name.strip() or "Monthly Plan", body.monthly_amount, body.years,
            body.fund_goal, body.fund_category, body.stock_goal, body.stock_universe,
        )
    return {"plan": _record_to_dict(record)}


@router.get("/saved")
async def list_saved_monthly_plans(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            "SELECT * FROM saved_monthly_plans WHERE user_id = $1::uuid ORDER BY created_at DESC",
            user_id,
        )
    return {"plans": [_record_to_dict(r) for r in records]}


@router.delete("/saved/{plan_id}")
async def delete_saved_monthly_plan(request: Request, plan_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM saved_monthly_plans WHERE id = $1 AND user_id = $2::uuid RETURNING id",
            plan_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Saved plan not found.")
    return {"ok": True}
