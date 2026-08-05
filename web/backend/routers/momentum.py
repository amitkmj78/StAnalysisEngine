import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.index_fund_service import get_index_fund_table
from services.momentum_backtest_service import (
    DEFAULT_BORROW_COST_BPS_ANNUAL,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SLIPPAGE_BPS,
    backtest_momentum_ranking,
)
from services.stock_finder_service import STOCK_UNIVERSES, get_stock_finder_table

from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/momentum",
    tags=["momentum"],
    dependencies=[Depends(verify_bearer_token)],
)

WINDOWS = {10, 30, 60, 90}
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]


@router.get("/options")
async def options():
    return {
        "windows": sorted(WINDOWS),
        "stock_universes": list(STOCK_UNIVERSES.keys()),
        "fund_categories": FUND_CATEGORIES,
    }


@router.get("/top-performers")
@limiter.limit("15/minute")
async def top_performers(
    request: Request,
    window: int = Query(30),
    asset_type: str = Query("Stock"),
    universe: str = Query("All"),
    top_n: int = Query(15, ge=1, le=50),
):
    await enforce_daily_quota(request, "momentum/top-performers")

    if window not in WINDOWS:
        raise HTTPException(422, f"window must be one of {sorted(WINDOWS)}")

    col = f"Return {window}D %"

    if asset_type == "Stock":
        if universe not in STOCK_UNIVERSES:
            raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")
        df = await run_in_threadpool(get_stock_finder_table, universe)
        name_col = "Name"
    elif asset_type == "Fund":
        df = await run_in_threadpool(get_index_fund_table)
        if universe != "All":
            if universe not in FUND_CATEGORIES:
                raise HTTPException(422, f"universe must be one of {FUND_CATEGORIES}")
            df = df[df["Category"] == universe]
        name_col = "Fund"
    else:
        raise HTTPException(422, "asset_type must be 'Stock' or 'Fund'")

    if df.empty or col not in df.columns:
        return {"results": [], "window": window, "asset_type": asset_type}

    ranked = df.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n)
    results = [
        {
            "ticker": str(row["Ticker"]),
            "name": str(row[name_col]),
            "price": round(float(row["Price"]), 2) if "Price" in row and row["Price"] == row["Price"] else None,
            "return_pct": round(float(row[col]), 2),
        }
        for _, row in ranked.iterrows()
    ]
    return {"results": results, "window": window, "asset_type": asset_type}


async def _persist_backtest_run(params: dict, result: dict) -> int:
    async with service_conn() as conn:
        run_id = await conn.fetchval(
            """
            INSERT INTO backtest_runs (
                asset_type, universe, lookback_days, top_n, years,
                slippage_bps, commission_bps, borrow_cost_bps_annual, risk_free_rate_annual,
                result_json
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """,
            params["asset_type"], params["universe"], params["lookback_days"], params["top_n"], params["years"],
            params["slippage_bps"], params["commission_bps"], params["borrow_cost_bps_annual"],
            params["risk_free_rate_annual"], json.dumps(result),
        )
    return run_id


@router.get("/backtest")
@limiter.limit("5/minute")
async def momentum_backtest(
    request: Request,
    asset_type: str = Query("Stock"),
    universe: str = Query("All"),
    lookback_days: int = Query(30),
    top_n: int = Query(5, ge=1, le=10),
    years: int = Query(3, ge=1, le=5),
    slippage_bps: float = Query(DEFAULT_SLIPPAGE_BPS, ge=0),
    commission_bps: float = Query(DEFAULT_COMMISSION_BPS, ge=0),
    borrow_cost_bps_annual: float = Query(DEFAULT_BORROW_COST_BPS_ANNUAL, ge=0),
    risk_free_rate_annual: float = Query(0.0, ge=0),
):
    """
    Story B / TR-7: walk-forward validation of the pure trailing-return
    ranking behind /top-performers, over `years` years — see
    momentum_backtest_service for why this (and not the fundamentals-
    weighted composite score) is the honest thing to backtest. Trading
    costs are applied by default (see the service module's documented
    defaults), and every run is persisted with its full parameter set,
    retrievable later by id via GET /momentum/backtest-runs/{id}.
    """
    await enforce_daily_quota(request, "momentum/backtest")

    if lookback_days not in WINDOWS:
        raise HTTPException(422, f"lookback_days must be one of {sorted(WINDOWS)}")
    if asset_type == "Stock":
        if universe not in STOCK_UNIVERSES:
            raise HTTPException(422, f"universe must be one of {sorted(STOCK_UNIVERSES.keys())}")
    elif asset_type == "Fund":
        if universe not in FUND_CATEGORIES:
            raise HTTPException(422, f"universe must be one of {FUND_CATEGORIES}")
    else:
        raise HTTPException(422, "asset_type must be 'Stock' or 'Fund'")

    result = await run_in_threadpool(
        backtest_momentum_ranking, asset_type, universe, lookback_days, top_n, years,
        slippage_bps, commission_bps, borrow_cost_bps_annual, risk_free_rate_annual,
    )
    if result is None:
        raise HTTPException(422, "Not enough historical data to run this backtest for the chosen settings.")

    params = {
        "asset_type": asset_type, "universe": universe, "lookback_days": lookback_days,
        "top_n": top_n, "years": years, "slippage_bps": slippage_bps, "commission_bps": commission_bps,
        "borrow_cost_bps_annual": borrow_cost_bps_annual, "risk_free_rate_annual": risk_free_rate_annual,
    }
    run_id = await _persist_backtest_run(params, result)

    return {"run_id": run_id, **result}


@router.get("/backtest-runs/{run_id}")
async def get_backtest_run(run_id: int):
    """TR-7: every run is re-fetchable by id, exactly as it was computed —
    no recomputation, so it stays correct even if price history or the
    model itself later changes."""
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM backtest_runs WHERE id = $1", run_id)
    if row is None:
        raise HTTPException(404, f"No backtest run with id {run_id}.")

    result = json.loads(row["result_json"]) if isinstance(row["result_json"], str) else row["result_json"]
    return {
        "run_id": row["id"],
        "requested_at_utc": row["requested_at_utc"].isoformat(),
        **result,
    }
