from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.million_plan_service import (
    ACCOUNT_TYPES,
    DEFAULT_TARGET_AMOUNT,
    DEFAULT_TARGET_YEARS,
    SOLVE_MODES,
    TAX_DRAG_PCT_BY_ACCOUNT,
    compute_goal_plan,
    get_diverse_strategy_picks,
)
from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS
from services.portfolio_performance_service import compute_total_portfolio_value
from services.stock_finder_service import STOCK_UNIVERSES
from services.strategy_plan_service import compute_plan_progress, elapsed_months

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/strategies",
    tags=["strategies"],
    dependencies=[Depends(verify_bearer_token)],
)

STOCK_GOALS = {"Short Term", "Long Term"}
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]
DOLLARS_MODES = ["today", "future"]


@router.get("/options")
async def options():
    return {
        "fund_goals": list(FUND_GOAL_WEIGHTS.keys()),
        "fund_categories": FUND_CATEGORIES,
        "stock_goals": list(STOCK_GOALS),
        "stock_universes": list(STOCK_UNIVERSES.keys()),
        "solve_modes": SOLVE_MODES,
        "account_types": ACCOUNT_TYPES,
        "tax_drag_pct_by_account": TAX_DRAG_PCT_BY_ACCOUNT,
        "defaults": {
            "target_amount": DEFAULT_TARGET_AMOUNT,
            "years": DEFAULT_TARGET_YEARS,
            "inflation_pct": 2.5,
        },
    }


@router.get("/summary")
@limiter.limit("10/minute")
async def summary(
    request: Request,
    mode: str = Query(...),
    target_amount: float | None = Query(None, ge=1, le=100_000_000),
    dollars_mode: str = Query("today"),
    years: float | None = Query(None, ge=0.1, le=20),
    starting_capital: float = Query(0, ge=0, le=10_000_000),
    monthly_contribution: float | None = Query(None, ge=0, le=1_000_000),
    annual_contribution_increase_pct: float = Query(0, ge=0, le=20),
    annual_return_pct: float | None = Query(None, ge=-20, le=50),
    inflation_pct: float = Query(2.5, ge=0, le=15),
    account_type: str = Query("Taxable"),
    fund_category: str = Query("All"),
    stock_universe: str = Query("All"),
    top_n: int = Query(1, ge=1, le=5),
):
    await enforce_daily_quota(request, "strategies/summary")

    if mode not in SOLVE_MODES:
        raise HTTPException(422, f"mode must be one of {SOLVE_MODES}")
    if dollars_mode not in DOLLARS_MODES:
        raise HTTPException(422, f"dollars_mode must be one of {DOLLARS_MODES}")
    if account_type not in ACCOUNT_TYPES:
        raise HTTPException(422, f"account_type must be one of {ACCOUNT_TYPES}")
    if fund_category not in FUND_CATEGORIES:
        raise HTTPException(422, f"fund_category must be one of {FUND_CATEGORIES}")
    if stock_universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"stock_universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    # Each solve mode needs everything except the one field it solves for --
    # validated here (422 with a clear message) rather than left to a
    # confusing downstream crash in compute_goal_plan.
    if mode != "achievable_amount" and target_amount is None:
        raise HTTPException(422, "target_amount is required unless mode is 'achievable_amount'.")
    if mode != "time_to_goal" and years is None:
        raise HTTPException(422, "years is required unless mode is 'time_to_goal'.")
    if mode != "required_contribution" and monthly_contribution is None:
        raise HTTPException(422, "monthly_contribution is required unless mode is 'required_contribution'.")
    if mode != "required_return" and annual_return_pct is None:
        raise HTTPException(422, "annual_return_pct is required unless mode is 'required_return'.")

    plan = await run_in_threadpool(
        compute_goal_plan,
        mode,
        target_amount,
        dollars_mode,
        years,
        starting_capital,
        monthly_contribution,
        annual_contribution_increase_pct,
        annual_return_pct,
        inflation_pct,
        account_type,
        fund_category,
        True,  # include_stock_picks -- this page always fans out to both funds and stocks
    )

    picks_out = None
    if plan.feasibility_level != "blocked":
        picks = await run_in_threadpool(get_diverse_strategy_picks, fund_category, stock_universe, top_n)
        picks_out = [asdict(pick) for pick in picks]

    return {"plan": asdict(plan), "picks": picks_out}


class SavePlanRequest(BaseModel):
    name: str | None = Field(default=None, max_length=100)
    target_amount: float = Field(ge=1, le=100_000_000)
    years: float = Field(ge=0.1, le=60)
    starting_capital: float = Field(ge=0, le=10_000_000)
    annual_return_pct: float = Field(ge=-50, le=100)
    monthly_contribution: float = Field(ge=0, le=1_000_000)
    annual_contribution_increase_pct: float = Field(default=0, ge=0, le=20)
    account_type: str = Field(default="Taxable")
    inflation_pct: float = Field(default=2.5, ge=0, le=15)


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
        "annual_contribution_increase_pct": row["annual_contribution_increase_pct"],
        "account_type": row["account_type"],
        "inflation_pct": row["inflation_pct"],
        "created_at": row["created_at"].isoformat(),
        "progress": progress,
    }


@router.post("/plans")
@limiter.limit("20/minute")
async def save_plan(body: SavePlanRequest, request: Request):
    """
    Stores the already-resolved monthly_contribution/annual_return_pct the
    client computed via /summary (whichever was given vs. solved for the
    chosen mode) rather than re-deriving a monthly figure server-side from a
    separate, older formula -- so a saved goal always matches what was on
    screen when it was saved, contribution growth/inflation/tax-drag
    included. Progress tracking still can't verify contributions were
    actually made (no ledger), same caveat as before.
    """
    await enforce_daily_quota(request, "strategies/plans/save")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO strategy_plans (
                user_id, name, target_amount, years, starting_capital, annual_return_pct, monthly_contribution,
                annual_contribution_increase_pct, account_type, inflation_pct
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            user_id, body.name, body.target_amount, body.years,
            body.starting_capital, body.annual_return_pct, body.monthly_contribution,
            body.annual_contribution_increase_pct, body.account_type, body.inflation_pct,
        )

    positions = await _user_positions(user_id)
    current_value = await run_in_threadpool(compute_total_portfolio_value, positions)
    progress = compute_plan_progress(
        starting_capital=row["starting_capital"],
        monthly_contribution=row["monthly_contribution"],
        annual_return_pct=row["annual_return_pct"],
        months_elapsed=elapsed_months(row["created_at"]),
        current_portfolio_value=current_value,
        annual_increase_pct=row["annual_contribution_increase_pct"],
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
            annual_increase_pct=row["annual_contribution_increase_pct"],
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
