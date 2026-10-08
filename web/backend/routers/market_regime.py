from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from services.market_overview_service import get_market_overview
from services.market_regime_service import (
    REGIME_GATE_DISCLOSURE,
    compute_and_persist_daily_regime,
    get_regime_snapshot,
)
from web.backend.admin import require_admin
from web.backend.app_settings import MARKET_REGIME_ENABLED_KEY, get_setting_bool
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn

router = APIRouter(prefix="/api/v1/market", tags=["market-regime"])


@router.get("/overview")
async def market_overview():
    """A small header ticker (S&P 500/Nasdaq/Dow) -- public, same
    posture as /stock/{ticker}/detail, server-cached 15 minutes (see
    services/market_overview_service.py) so this is cheap regardless
    of how many open tabs/users poll it."""
    return {"indices": await run_in_threadpool(get_market_overview)}


@router.get("/regime", dependencies=[Depends(verify_bearer_token)])
async def get_market_regime():
    """REG-1: the current site-wide regime reading for RegimeBanner.tsx.
    Returns {"available": false, "reason": ...} rather than a 404/500
    when the feature is off or hasn't produced a row yet — see
    services/market_regime_service.py's module docstring for why this
    ships despite a failed validation gate, and REGIME_GATE_DISCLOSURE
    for the disclosure every response below carries."""
    if not await get_setting_bool(MARKET_REGIME_ENABLED_KEY, default=False):
        return {"available": False, "reason": "market_regime_enabled is off"}

    snapshot = await get_regime_snapshot()
    if snapshot is None:
        return {"available": False, "reason": "no regime data has been computed yet"}

    return {"available": True, **snapshot}


@router.get("/regime-history", dependencies=[Depends(verify_bearer_token)])
async def get_regime_history(days: int = Query(400, ge=30, le=1500)):
    """DIF-9: the stored regime label for each trading day, for shading the price chart.
    Same switch as /regime. The label is a condition label, not a recommendation."""
    if not await get_setting_bool(MARKET_REGIME_ENABLED_KEY, default=False):
        return {"available": False, "reason": "market_regime_enabled is off", "history": [], "disclosure": REGIME_GATE_DISCLOSURE}
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT as_of_date, regime_confirmed FROM market_regime_daily
            WHERE as_of_date >= CURRENT_DATE - $1::int ORDER BY as_of_date
            """,
            days,
        )
    return {
        "available": True,
        "history": [{"date": r["as_of_date"].isoformat(), "regime": r["regime_confirmed"]} for r in rows],
        "disclosure": REGIME_GATE_DISCLOSURE,
    }


@router.post("/regime/backfill", dependencies=[Depends(require_admin)])
async def backfill_market_regime():
    """One-time (or occasional re-run) historical population of
    market_regime_daily across the full fetched history window, so REG-2's
    by-regime track record has real depth instead of accumulating for
    weeks. Manual only — never called by the scheduler, app boot, or the
    DDL migration itself, since it fans out ~500+8 tickers' full price
    history (see services/market_data_service.py::fetch_market_internals_history).
    Safe to re-run: every write is ON CONFLICT (as_of_date) DO UPDATE."""
    result = await compute_and_persist_daily_regime(backfill=True)
    return result
