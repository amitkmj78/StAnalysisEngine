from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.million_plan_service import (
    DEFAULT_TARGET_AMOUNT,
    DEFAULT_TARGET_YEARS,
    build_million_plan_table_from_returns,
    get_diverse_strategy_picks,
    project_total_future_value,
    required_monthly_investment,
)
from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS
from services.portfolio_performance_service import compute_total_portfolio_value
from services.stock_finder_service import STOCK_UNIVERSES
from services.strategy_plan_service import compute_plan_progress, elapsed_months

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/strategies",
    tags=["strategies"],
    dependencies=[Depends(verify_bearer_token)],
)

STOCK_GOALS = {"Short Term", "Long Term"}
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]


@router.get("/options")
async def options():
    return {
        "fund_goals": list(FUND_GOAL_WEIGHTS.keys()),
        "fund_categories": FUND_CATEGORIES,
        "stock_goals": list(STOCK_GOALS),
        "stock_universes": list(STOCK_UNIVERSES.keys()),
        "defaults": {"target_amount": DEFAULT_TARGET_AMOUNT, "years": DEFAULT_TARGET_YEARS},
    }


@router.get("/summary")
@limiter.limit("10/minute")
async def summary(
    request: Request,
    target_amount: float = Query(DEFAULT_TARGET_AMOUNT, ge=50_000, le=10_000_000),
    years: int = Query(DEFAULT_TARGET_YEARS, ge=1, le=20),
    starting_capital: float = Query(0, ge=0, le=10_000_000),
    min_return: int = Query(6, ge=4, le=18),
    max_return: int = Query(15, ge=5, le=20),
    return_step: int = Query(2),
    custom_return: float = Query(10, ge=4, le=20),
    fund_category: str = Query("All"),
    stock_universe: str = Query("All"),
    top_n: int = Query(1, ge=1, le=5),
):
    await enforce_daily_quota(request, "strategies/summary")

    if return_step not in (1, 2, 3):
        raise HTTPException(422, "return_step must be 1, 2, or 3")
    if max_return <= min_return:
        raise HTTPException(422, "max_return must be greater than min_return")
    if fund_category not in FUND_CATEGORIES:
        raise HTTPException(422, f"fund_category must be one of {FUND_CATEGORIES}")
    if stock_universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"stock_universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    return_cases = list(range(min_return, max_return + 1, return_step))
    if float(custom_return) not in return_cases:
        return_cases.append(float(custom_return))

    plan_df = await run_in_threadpool(
        build_million_plan_table_from_returns,
        return_cases,
        target_amount,
        years,
        starting_capital,
    )
    custom_monthly = required_monthly_investment(
        target_amount=target_amount,
        years=years,
        annual_return_pct=custom_return,
        starting_capital=starting_capital,
    )

    picks = await run_in_threadpool(
        get_diverse_strategy_picks, fund_category, stock_universe, top_n
    )

    picks_out = []
    for pick in picks:
        implied_monthly = None
        projected_value = None
        if pick.annual_return_pct is not None:
            implied_monthly = required_monthly_investment(
                target_amount=target_amount,
                years=years,
                annual_return_pct=pick.annual_return_pct,
                starting_capital=starting_capital,
            )
            projected_value = project_total_future_value(
                monthly_amount=implied_monthly,
                years=years,
                annual_return_pct=pick.annual_return_pct,
                starting_capital=starting_capital,
            )
        picks_out.append(
            {
                **asdict(pick),
                "implied_monthly": implied_monthly,
                "projected_value": projected_value,
            }
        )

    return {
        "plan_table": records_safe(plan_df),
        "custom_monthly": custom_monthly,
        "picks": picks_out,
    }


class SavePlanRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    target_amount: float = Field(ge=50_000, le=10_000_000)
    years: int = Field(ge=1, le=20)
    starting_capital: float = Field(ge=0, le=10_000_000)
    annual_return_pct: float = Field(ge=-50, le=100)


async def _user_positions(user_id: str) -> list[dict]:
    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid",
            user_id,
        )
    return [{"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"]} for r in records]


def _plan_out(row, progress: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "target_amount": row["target_amount"],
        "years": row["years"],
        "starting_capital": row["starting_capital"],
        "annual_return_pct": row["annual_return_pct"],
        "monthly_contribution": row["monthly_contribution"],
        "created_at": row["created_at"].isoformat(),
        "progress": progress,
    }


@router.post("/plans")
@limiter.limit("20/minute")
async def save_plan(body: SavePlanRequest, request: Request):
    """Locks in the required-monthly-contribution math server-side (never
    trusts a client-submitted monthly figure) so progress tracking later
    compares against a number the user couldn't have fudged upward to make
    themselves look ahead of pace."""
    await enforce_daily_quota(request, "strategies/plans/save")
    user_id = request.state.user["id"]

    monthly_contribution = required_monthly_investment(
        target_amount=body.target_amount,
        years=body.years,
        annual_return_pct=body.annual_return_pct,
        starting_capital=body.starting_capital,
    )

    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO strategy_plans (
                user_id, name, target_amount, years, starting_capital, annual_return_pct, monthly_contribution
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            user_id, body.name, body.target_amount, body.years,
            body.starting_capital, body.annual_return_pct, monthly_contribution,
        )

    positions = await _user_positions(user_id)
    current_value = await run_in_threadpool(compute_total_portfolio_value, positions)
    progress = compute_plan_progress(
        starting_capital=row["starting_capital"],
        monthly_contribution=row["monthly_contribution"],
        annual_return_pct=row["annual_return_pct"],
        months_elapsed=elapsed_months(row["created_at"]),
        current_portfolio_value=current_value,
    )
    return _plan_out(row, progress)


@router.get("/plans")
async def list_plans(request: Request):
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        rows = await conn.fetch("SELECT * FROM strategy_plans ORDER BY created_at DESC")

    if not rows:
        return {"plans": []}

    positions = await _user_positions(user_id)
    current_value = await run_in_threadpool(compute_total_portfolio_value, positions)

    plans = []
    for row in rows:
        progress = compute_plan_progress(
            starting_capital=row["starting_capital"],
            monthly_contribution=row["monthly_contribution"],
            annual_return_pct=row["annual_return_pct"],
            months_elapsed=elapsed_months(row["created_at"]),
            current_portfolio_value=current_value,
        )
        plans.append(_plan_out(row, progress))

    return {"plans": plans}


@router.delete("/plans/{plan_id}")
async def delete_plan(plan_id: int, request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM strategy_plans WHERE id = $1 AND user_id = $2::uuid RETURNING id",
            plan_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Plan not found.")
    return {"ok": True}
