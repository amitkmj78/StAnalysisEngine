from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.baseline_service import HORIZONS, InsufficientHistoryError, compute_baseline_band
from services.data_service import get_adjusted_history

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/baseline",
    tags=["baseline"],
    dependencies=[Depends(verify_bearer_token)],
)


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


# --- Save a band snapshot now, so it can be compared against a freshly
# recomputed one later (e.g. after price has moved, or history has grown).
# Registered before GET /{ticker} below so "/save" and "/history" aren't
# swallowed by that catch-all path parameter.

class SaveBaselineRequest(BaseModel):
    ticker: str
    horizon_days: int
    confidence: float
    method: str
    as_of: str
    last_price: float
    floor: float
    floor_pct: float
    accumulation_zone_hi: float
    accumulation_zone_hi_pct: float
    median_path: float
    distribution_zone_lo: float
    distribution_zone_lo_pct: float
    ceiling: float
    ceiling_pct: float
    samples: int
    effective_samples: int
    breach_rate_full: float


@router.post("/save")
@limiter.limit("20/minute")
async def save_baseline_snapshot(request: Request, body: SaveBaselineRequest):
    await enforce_daily_quota(request, "baseline/save")
    user_id = request.state.user["id"]
    ticker = body.ticker.strip().upper()

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO saved_baseline_snapshots (
                user_id, ticker, horizon_days, confidence, method, as_of, last_price,
                floor, floor_pct, accumulation_zone_hi, accumulation_zone_hi_pct,
                median_path, distribution_zone_lo, distribution_zone_lo_pct,
                ceiling, ceiling_pct, samples, effective_samples, breach_rate_full
            ) VALUES (
                $1::uuid, $2, $3, $4, $5, $6::date, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
            )
            RETURNING *
            """,
            user_id, ticker, body.horizon_days, body.confidence, body.method, body.as_of, body.last_price,
            body.floor, body.floor_pct, body.accumulation_zone_hi, body.accumulation_zone_hi_pct,
            body.median_path, body.distribution_zone_lo, body.distribution_zone_lo_pct,
            body.ceiling, body.ceiling_pct, body.samples, body.effective_samples, body.breach_rate_full,
        )
    return {"snapshot": _record_to_dict(record)}


@router.get("/history")
async def baseline_history(request: Request, ticker: str | None = Query(None)):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        if ticker:
            records = await conn.fetch(
                "SELECT * FROM saved_baseline_snapshots WHERE ticker = $1 ORDER BY saved_at DESC",
                ticker.strip().upper(),
            )
        else:
            records = await conn.fetch("SELECT * FROM saved_baseline_snapshots ORDER BY saved_at DESC")
    return {"snapshots": [_record_to_dict(r) for r in records]}


@router.delete("/snapshot/{snapshot_id}")
async def delete_baseline_snapshot(request: Request, snapshot_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM saved_baseline_snapshots WHERE id = $1 RETURNING id",
            snapshot_id,
        )
    if row is None:
        raise HTTPException(404, "Saved snapshot not found.")
    return {"ok": True}


@router.get("/{ticker}")
@limiter.limit("30/minute")
async def get_baseline(
    request: Request,
    ticker: str,
    horizon: int = Query(30),
    confidence: float = Query(0.90),
    method: str = Query("empirical"),
    half_life: int | None = Query(126),
):
    await enforce_daily_quota(request, "baseline/get")

    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {HORIZONS}")
    if not (0.5 <= confidence < 1.0):
        raise HTTPException(400, "confidence must be in [0.5, 1.0)")
    if method not in ("empirical", "sqrt"):
        raise HTTPException(400, "method must be 'empirical' or 'sqrt'")
    if half_life is not None and half_life <= 0:
        raise HTTPException(400, "half_life must be > 0 or null")

    cleaned = ticker.strip().upper()
    df = await run_in_threadpool(get_adjusted_history, cleaned, "3y")
    if df.empty:
        raise HTTPException(404, f"Unknown ticker: {cleaned}")

    try:
        result = await run_in_threadpool(
            compute_baseline_band, df, horizon, confidence, method, half_life
        )
    except InsufficientHistoryError as e:
        raise HTTPException(
            422,
            {
                "message": str(e),
                "bars_required": e.bars_required,
                "bars_available": e.bars_available,
            },
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"ticker": cleaned, **result}
