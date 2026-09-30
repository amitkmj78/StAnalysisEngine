from fastapi import APIRouter, Depends

from services.market_regime_service import compute_and_persist_daily_regime, get_regime_snapshot
from web.backend.admin import require_admin
from web.backend.app_settings import MARKET_REGIME_ENABLED_KEY, get_setting_bool
from web.backend.auth import verify_bearer_token

router = APIRouter(prefix="/api/v1/market", tags=["market-regime"])


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
