from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.entry_strategy_service import (
    ENTRY_FUND_UNIVERSES,
    ENTRY_STOCK_UNIVERSES,
    build_entry_plan,
    scan_best_entries,
)

from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
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


QUANT_SIGNALS = {"BUY", "HOLD", "SELL"}


async def _attach_quant_signals(results: list[dict]) -> None:
    """
    Joins each scan row with the pre-captured Quant Signal (BUY/HOLD/SELL
    from pit_quant_signal, the same daily point-in-time snapshot the
    Stock Screener and /quant-vs-analyst use) — one bulk query regardless
    of how many rows (an indexed WHERE ticker = ANY(...) over the full
    scanned universe costs about the same as over just top_n), not a
    live per-ticker forecast, since scanning up to 500 tickers already
    takes ~20s on its own. Called against the *full* scanned set, before
    any quant_signal filter/top_n truncation, so "show only BUY" isn't
    limited to whatever happened to already be in the technical top_n.
    A ticker missing from that day's capture (not an S&P 500 member, or
    a capture gap) just gets nulls, mutated in place rather than
    dropped, matching /quant-vs-analyst's existing "shown as missing,
    not hidden" precedent for this same join.
    """
    tickers = [r["Ticker"] for r in results]
    if not tickers:
        return

    async with service_conn() as conn:
        as_of_date = await conn.fetchval("SELECT max(as_of_date) FROM pit_quant_signal")
        rows = []
        if as_of_date is not None:
            rows = await conn.fetch(
                """
                SELECT ticker, signal, expected_return_pct, target_price
                FROM pit_quant_signal
                WHERE as_of_date = $1 AND ticker = ANY($2::text[])
                """,
                as_of_date,
                tickers,
            )

    by_ticker = {r["ticker"]: r for r in rows}
    for row in results:
        q = by_ticker.get(row["Ticker"])
        row["Quant Signal"] = q["signal"] if q else None
        row["Quant Expected Return %"] = float(q["expected_return_pct"]) if q else None
        row["Quant Target Price"] = float(q["target_price"]) if q else None


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
    quant_signal: str | None = Query(None),
):
    await enforce_daily_quota(request, "entry/scan")
    _validate_asset_type(asset_type)
    valid_universes = _universes_for(asset_type)
    if universe not in valid_universes:
        raise HTTPException(422, f"universe must be one of {sorted(valid_universes.keys())}")
    if quant_signal is not None and quant_signal not in QUANT_SIGNALS:
        raise HTTPException(422, f"quant_signal must be one of {sorted(QUANT_SIGNALS)}")
    if quant_signal is not None and asset_type != "Stock":
        raise HTTPException(422, "quant_signal filtering is only available for asset_type=Stock.")

    df = await run_in_threadpool(scan_best_entries, asset_type, universe)
    results = records_safe(df)  # full scanned universe, still in Entry Score order
    if asset_type == "Stock" and results:
        await _attach_quant_signals(results)
    if quant_signal is not None:
        results = [r for r in results if r.get("Quant Signal") == quant_signal]
    return {"results": results[:top_n]}


@router.get("/plan")
@limiter.limit("20/minute")
async def plan(request: Request, ticker: str = Query(..., min_length=1)):
    await enforce_daily_quota(request, "entry/plan")
    ticker = ticker.strip().upper()

    result = await run_in_threadpool(build_entry_plan, ticker, True)
    if result is None:
        return {"plan": None, "history": None}

    history_df = result.pop("history").tail(120)
    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in history_df.index],
        "close": history_df["Close"].round(4).tolist(),
    }
    return {"plan": result, "history": history}
