"""
Scenario and Stress Tests (docs/stock-analysis-requirements.html, STR-1..3).
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.portfolio_review_service import compute_sectors
from services.stress_test_service import (
    HISTORICAL_REPLAYS,
    SHOCK_PRESETS,
    run_custom_scenario,
    run_historical_replay,
    run_preset_shock,
)
from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _eastern_today, _resolve_portfolio_id

router = APIRouter(
    prefix="/api/v1/portfolio/stress-test",
    tags=["stress-test"],
    dependencies=[Depends(verify_bearer_token)],
)


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


def _scenario_to_dict(record) -> dict:
    row = _record_to_dict(record)
    if isinstance(row.get("shock_config"), str):
        row["shock_config"] = json.loads(row["shock_config"])
    return row


async def _load_positions(request: Request, portfolio_id: Optional[int]) -> list[dict]:
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    return [
        {"ticker": r["ticker"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]


@router.get("/presets")
def list_presets():
    """Static catalog -- no auth-scoped data, just what the UI renders
    buttons from. Kept under the authenticated prefix anyway for
    consistency with every other /api/v1/portfolio/* route."""
    return {
        "presets": [
            {
                "preset_key": k, "label": v["label"], "benchmark_ticker": v["benchmark_ticker"],
                "shock_pct": v["shock_pct"], "method": v["method"],
            }
            for k, v in SHOCK_PRESETS.items()
        ],
        "replays": [
            {
                "replay_key": k, "label": v["label"], "window_start": v["start"],
                "window_end": v["end"], "method": v["method"],
            }
            for k, v in HISTORICAL_REPLAYS.items()
        ],
    }


@router.get("/run")
@limiter.limit("10/minute")
async def run_stress_test(
    request: Request, kind: str, key: str, portfolio_id: Optional[int] = None
):
    """STR-1: kind="preset" (beta-based shock) or kind="replay" (actual
    historical return)."""
    await enforce_daily_quota(request, "portfolio/stress-test/run")
    if kind not in ("preset", "replay"):
        raise HTTPException(422, "kind must be 'preset' or 'replay'.")
    if kind == "preset" and key not in SHOCK_PRESETS:
        raise HTTPException(404, f"Unknown preset '{key}'.")
    if kind == "replay" and key not in HISTORICAL_REPLAYS:
        raise HTTPException(404, f"Unknown replay '{key}'.")

    positions = await _load_positions(request, portfolio_id)
    if kind == "preset":
        result = await run_in_threadpool(run_preset_shock, positions, key)
    else:
        result = await run_in_threadpool(run_historical_replay, positions, key)
    return {"as_of": str(_eastern_today()), "result": result}


class ScenarioComponent(BaseModel):
    kind: str  # "sector" | "factor"
    label: str
    sector: Optional[str] = None
    benchmark_ticker: Optional[str] = None
    shock_pct: float


class RunCustomScenarioRequest(BaseModel):
    components: list[ScenarioComponent]
    portfolio_id: Optional[int] = None


class SaveScenarioRequest(BaseModel):
    name: str
    shock_config: list[ScenarioComponent]


@router.post("/custom")
@limiter.limit("10/minute")
async def run_custom(request: Request, body: RunCustomScenarioRequest):
    """STR-2: a user-built combination of sector and/or factor moves --
    see services.stress_test_service.run_custom_scenario for exactly how
    each component kind is computed and why they're combined by simple
    addition (disclosed in the result's own method field, not hidden)."""
    await enforce_daily_quota(request, "portfolio/stress-test/custom")
    positions = await _load_positions(request, body.portfolio_id)
    tickers = [p["ticker"] for p in positions]
    sector_by_ticker = await run_in_threadpool(compute_sectors, tickers)
    result = await run_in_threadpool(
        run_custom_scenario, positions, sector_by_ticker, [c.model_dump() for c in body.components],
    )
    return {"as_of": str(_eastern_today()), "result": result}


@router.post("/scenarios")
@limiter.limit("20/minute")
async def save_scenario(request: Request, body: SaveScenarioRequest):
    """STR-2: "custom scenario saves and reruns" -- rerun is entirely
    client-side (the frontend reloads shock_config into the builder's
    own state and calls /custom again), same precedent as Stock Finder's
    saved screens (web/frontend/app/stock-finder/page.tsx::
    handleLoadScreen) -- no backend rerun endpoint needed."""
    await enforce_daily_quota(request, "portfolio/stress-test/scenarios/save")
    user_id = request.state.user["id"]
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "name is required")
    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            "INSERT INTO saved_stress_scenarios (user_id, name, shock_config) VALUES ($1::uuid, $2, $3::jsonb) RETURNING *",
            user_id, name, json.dumps([c.model_dump() for c in body.shock_config]),
        )
    return {"scenario": _scenario_to_dict(record)}


@router.get("/scenarios")
async def list_scenarios(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM saved_stress_scenarios ORDER BY saved_at DESC")
    return {"scenarios": [_scenario_to_dict(r) for r in records]}


@router.delete("/scenarios/{scenario_id}")
async def delete_scenario(request: Request, scenario_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("DELETE FROM saved_stress_scenarios WHERE id = $1 RETURNING id", scenario_id)
    if row is None:
        raise HTTPException(404, "Saved scenario not found.")
    return {"ok": True}
