import json
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.fund_comparison_service import price_near_date, rank_funds_by_inception
from services.index_fund_service import (
    CUSTOM_WEIGHTABLE_METRICS,
    GOAL_WEIGHTS,
    InvalidCustomWeights,
    LOWER_IS_BETTER,
    METRIC_LABELS,
    VALID_WINDOWS,
    normalize_custom_weights,
    rank_index_funds,
    score_fund_ticker,
)

from web.backend.auth import verify_bearer_token
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.utils import records_safe

router = APIRouter(
    prefix="/api/v1/index-fund",
    tags=["index-fund"],
    dependencies=[Depends(verify_bearer_token)],
)

FUND_CATEGORIES = [
    "All",
    "US Large Blend",
    "US Total Market",
    "US Large Growth",
    "US Large Value",
    "US Mid Cap",
    "US Small Cap",
    "International Developed",
    "International Total",
    "Emerging Markets",
    "Bond — Total Market",
    "Bond — Short-Term",
    "Bond — Long-Term/Treasury",
    "Bond — Corporate",
    "Bond — High Yield",
    "Bond — TIPS",
    "Dividend/Income",
    "Real Estate",
    "Sector",
]


def _validate_goal(goal: str) -> None:
    if goal not in GOAL_WEIGHTS and goal != "Custom":
        raise HTTPException(422, f"goal must be one of {sorted(GOAL_WEIGHTS.keys())} or 'Custom'")


def _validate_window(window: str) -> None:
    if window not in VALID_WINDOWS:
        raise HTTPException(422, f"window must be one of {sorted(VALID_WINDOWS)}")


def _parse_custom_weights(goal: str, weights_json: str | None) -> dict[str, float] | None:
    """Only consulted when goal == "Custom". Parses the JSON query param and
    delegates validation/normalization to services.index_fund_service's
    normalize_custom_weights (kept dependency-free there so it can be unit
    tested without pulling in FastAPI/slowapi) -- this function's only job
    is translating that into an HTTP 422."""
    if goal != "Custom":
        return None
    if not weights_json:
        raise HTTPException(422, "weights is required when goal is 'Custom'.")
    try:
        raw = json.loads(weights_json)
    except (TypeError, ValueError):
        raise HTTPException(422, "weights must be valid JSON.")
    if not isinstance(raw, dict):
        raise HTTPException(422, "weights must be a JSON object of metric -> weight.")
    try:
        return normalize_custom_weights(raw)
    except InvalidCustomWeights as exc:
        raise HTTPException(422, str(exc))


@router.get("/goals")
async def goals():
    """Each goal's real weights, with display labels -- so the frontend can
    show them inline (FS-4) without a second endpoint. The four presets'
    weights are fixed; "Custom" carries the full weightable-metric list
    instead, for the slider UI to build itself from."""
    return {
        "goals": [
            {
                "name": name,
                "weights": [
                    {
                        "metric": metric,
                        "label": METRIC_LABELS.get(metric, metric),
                        "weight": weight,
                        "lower_is_better": metric in LOWER_IS_BETTER,
                    }
                    for metric, weight in weights.items()
                ],
            }
            for name, weights in GOAL_WEIGHTS.items()
        ]
        + [
            {
                "name": "Custom",
                "weights": [
                    {
                        "metric": metric,
                        "label": METRIC_LABELS.get(metric, metric),
                        "weight": None,
                        "lower_is_better": metric in LOWER_IS_BETTER,
                    }
                    for metric in CUSTOM_WEIGHTABLE_METRICS
                ],
            }
        ]
    }


@router.get("/categories")
async def categories():
    return {"categories": FUND_CATEGORIES}


@router.get("/windows")
async def windows():
    return {"windows": sorted(VALID_WINDOWS, key=lambda w: (w != "max_common", w))}


@router.get("/rank")
@limiter.limit("10/minute")
async def rank(
    request: Request,
    goal: str = Query(...),
    category: str = Query("All"),
    window: str = Query("5y"),
    weights: str | None = Query(None, description="JSON metric->weight, required when goal='Custom'"),
):
    await enforce_daily_quota(request, "index-fund/rank")
    _validate_goal(goal)
    _validate_window(window)
    if category not in FUND_CATEGORIES:
        raise HTTPException(422, f"category must be one of {FUND_CATEGORIES}")
    custom_weights = _parse_custom_weights(goal, weights)

    df, window_meta = await run_in_threadpool(rank_index_funds, goal, category, window, custom_weights)
    return {"results": records_safe(df), **window_meta}


@router.get("/rank-by-inception")
@limiter.limit("10/minute")
async def rank_by_inception(
    request: Request,
    min_years: int = Query(..., ge=1, le=50),
    category: str = Query("All"),
):
    """
    Funds with at least `min_years` of real trading history, ranked by
    their real since-inception % return — a long-run track record view.
    Same cost profile as /rank (built on the same cached fund table),
    plus one "max"-period price lookup per surviving fund, so same
    quota tier.
    """
    await enforce_daily_quota(request, "index-fund/rank-by-inception")
    if category not in FUND_CATEGORIES:
        raise HTTPException(422, f"category must be one of {FUND_CATEGORIES}")

    df = await run_in_threadpool(rank_funds_by_inception, min_years, category)
    return {"results": records_safe(df)}


@router.get("/return-since")
@limiter.limit("20/minute")
async def return_since(request: Request, ticker: str = Query(..., min_length=1), since: date = Query(...)):
    """
    Real, point-in-time return for one ticker from `since` to now — "what
    if you'd put this money in this fund instead, starting the same day,"
    not a fixed 30d/1Y/3Y window that may not match how long the caller has
    actually been invested. `since` can be decades back (e.g. a fund's
    inception date) — price_near_date is asked for "max" history, not the
    2y default, to cover that.
    """
    await enforce_daily_quota(request, "index-fund/return-since")
    ticker = ticker.strip().upper()
    since_dt = datetime.combine(since, datetime.min.time())

    price_then = await run_in_threadpool(price_near_date, ticker, since_dt, "max")
    price_now = await run_in_threadpool(get_latest_price, ticker)
    if price_then is None or price_now is None:
        raise HTTPException(404, f"No price history found for {ticker}.")

    return {
        "ticker": ticker,
        "since": str(since),
        "days": (date.today() - since).days,
        "price_then": round(price_then, 2),
        "price_now": round(price_now, 2),
        "return_pct": round((price_now - price_then) / price_then * 100, 2) if price_then else None,
    }


@router.get("/score")
@limiter.limit("20/minute")
async def score(
    request: Request,
    goal: str = Query(...),
    ticker: str = Query(..., min_length=1),
    window: str = Query("5y"),
    weights: str | None = Query(None, description="JSON metric->weight, required when goal='Custom'"),
):
    await enforce_daily_quota(request, "index-fund/score")
    _validate_goal(goal)
    _validate_window(window)
    ticker = ticker.strip().upper()
    custom_weights = _parse_custom_weights(goal, weights)

    df, window_meta = await run_in_threadpool(score_fund_ticker, goal, ticker, window, custom_weights)
    records = records_safe(df)
    return {"result": records[0] if records else None, **window_meta}
