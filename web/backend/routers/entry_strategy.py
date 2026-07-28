from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.entry_strategy_service import (
    ENTRY_FUND_UNIVERSES,
    ENTRY_STOCK_UNIVERSES,
    build_entry_plan,
    scan_best_entries,
)

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/entry",
    tags=["entry"],
    dependencies=[Depends(verify_bearer_token)],
)

ASSET_TYPES = {"Fund", "Stock"}


def _universes_for(asset_type: str):
    return ENTRY_FUND_UNIVERSES if asset_type == "Fund" else ENTRY_STOCK_UNIVERSES


def _validate_asset_type(asset_type: str) -> None:
    if asset_type not in ASSET_TYPES:
        raise HTTPException(422, f"asset_type must be one of {sorted(ASSET_TYPES)}")


@router.get("/universes")
async def universes(asset_type: str = Query(...)):
    _validate_asset_type(asset_type)
    return {"universes": list(_universes_for(asset_type).keys())}


@router.get("/scan")
@limiter.limit("10/minute")
async def scan(
    request: Request,
    asset_type: str = Query(...),
    universe: str = Query("All"),
    top_n: int = Query(5, ge=1, le=20),
):
    await enforce_daily_quota(request, "entry/scan")
    _validate_asset_type(asset_type)
    valid_universes = _universes_for(asset_type)
    if universe not in valid_universes:
        raise HTTPException(422, f"universe must be one of {sorted(valid_universes.keys())}")

    df = await run_in_threadpool(scan_best_entries, asset_type, universe)
    if not df.empty:
        df = df.head(top_n)
    return {"results": records_safe(df)}


@router.get("/plan")
@limiter.limit("20/minute")
async def plan(request: Request, ticker: str = Query(..., min_length=1)):
    await enforce_daily_quota(request, "entry/plan")
    ticker = ticker.strip().upper()

    result = await run_in_threadpool(build_entry_plan, ticker)
    if result is None:
        return {"plan": None, "history": None}

    history_df = result.pop("history").tail(120)
    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in history_df.index],
        "close": history_df["Close"].round(4).tolist(),
    }
    return {"plan": result, "history": history}
