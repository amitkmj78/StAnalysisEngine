from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.index_fund_service import get_index_fund_table
from services.momentum_backtest_service import backtest_momentum_ranking
from services.stock_finder_service import STOCK_UNIVERSES, get_stock_finder_table

from web.backend.auth import verify_bearer_token
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
            "return_pct": round(float(row[col]), 2),
        }
        for _, row in ranked.iterrows()
    ]
    return {"results": results, "window": window, "asset_type": asset_type}


@router.get("/backtest")
@limiter.limit("5/minute")
async def momentum_backtest(
    request: Request,
    asset_type: str = Query("Stock"),
    universe: str = Query("All"),
    lookback_days: int = Query(30),
    top_n: int = Query(5, ge=1, le=10),
    years: int = Query(3, ge=1, le=5),
):
    """
    Story B: walk-forward validation of the pure trailing-return ranking
    behind /top-performers, over `years` years — see
    momentum_backtest_service for why this (and not the fundamentals-
    weighted composite score) is the honest thing to backtest.
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
        backtest_momentum_ranking, asset_type, universe, lookback_days, top_n, years
    )
    if result is None:
        raise HTTPException(422, "Not enough historical data to run this backtest for the chosen settings.")
    return result
