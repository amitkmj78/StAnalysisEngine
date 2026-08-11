from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.baseline_service import HORIZONS, InsufficientHistoryError, compute_baseline_band
from services.data_service import get_adjusted_history

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(
    prefix="/api/v1/baseline",
    tags=["baseline"],
    dependencies=[Depends(verify_bearer_token)],
)


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
