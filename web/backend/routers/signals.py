from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from services.signal_publication_service import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PREDICT_PERIOD,
    DEFAULT_UNIVERSE,
    PREDICT_COMPARE_HORIZONS,
    compute_outcome_metrics,
    compute_predict_algo_comparison,
)
from web.backend.admin import require_admin
from web.backend.app_settings import PUBLISH_SIGNALS_ENABLED_KEY, get_setting_bool
from web.backend.db import service_conn
from web.backend.rate_limit import limiter
from web.backend.signal_publication import evaluate_due_signal_outcomes, publish_daily_signals

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


@router.get("/published")
@limiter.limit("60/minute")
async def list_published_signals(
    request: Request,
    target_date: date | None = Query(None),
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
):
    """
    TR-5 groundwork: the public track record, unauthenticated by design —
    this is the whole point of "published". Defaults to the most recent
    publication for the given universe/lookback when no date is given.
    """
    async with service_conn() as conn:
        summary = await conn.fetchrow(
            """
            SELECT min(target_date) AS record_start_date,
                   max(target_date) AS latest_date,
                   count(DISTINCT target_date) AS days_published
            FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            """,
            universe_id, lookback_days,
        )
        record_start_date = summary["record_start_date"] if summary else None
        days_published = summary["days_published"] if summary else 0

        if target_date is None:
            target_date = summary["latest_date"] if summary else None
            if target_date is None:
                return {
                    "target_date": None,
                    "universe_id": universe_id,
                    "lookback_days": lookback_days,
                    "signals": [],
                    "record_start_date": None,
                    "days_published": 0,
                }

        rows = await conn.fetch(
            """
            SELECT * FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            target_date, universe_id, lookback_days,
        )

    return {
        "target_date": str(target_date),
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "signals": [_record_to_dict(r) for r in rows],
        "record_start_date": str(record_start_date) if record_start_date else None,
        "days_published": days_published,
    }


@router.get("/published/compare-to-predict-algo")
@limiter.limit("10/minute")
async def compare_to_predict_algo(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    days_ahead: int = Query(DEFAULT_LOOKBACK_DAYS),
):
    """
    What the separate, trained Predict-page model currently says about
    today's published momentum picks — a different algorithm, not a
    validation of this one. Deliberately only ever compares against the
    *latest* publication (no target_date param): running today's model
    against an older published date would mix in price data the model
    couldn't have had at that original date, since there's no point-in-time
    store yet to prevent that honestly.

    days_ahead is the Predict-algo forecast horizon, restricted to
    PREDICT_COMPARE_HORIZONS (1/5/10/30) so short reads (does the algorithm
    agree over the next day or week?) are possible too, not just the
    30-day window matching the momentum lookback.
    """
    if days_ahead not in PREDICT_COMPARE_HORIZONS:
        raise HTTPException(422, f"days_ahead must be one of {PREDICT_COMPARE_HORIZONS}")
    predict_days_ahead = days_ahead
    async with service_conn() as conn:
        latest_date = await conn.fetchval(
            """
            SELECT max(target_date) FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            """,
            universe_id, lookback_days,
        )
        if latest_date is None:
            return {"target_date": None, "comparisons": []}

        rows = await conn.fetch(
            """
            SELECT ticker, rank, trailing_return_pct FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            latest_date, universe_id, lookback_days,
        )

    tickers = [r["ticker"] for r in rows]
    comparison = await run_in_threadpool(
        compute_predict_algo_comparison, tickers, DEFAULT_PREDICT_PERIOD, predict_days_ahead
    )
    comparison_by_ticker = {c["ticker"]: c for c in comparison}

    return {
        "target_date": str(latest_date),
        "predict_period": DEFAULT_PREDICT_PERIOD,
        "predict_days_ahead": predict_days_ahead,
        "comparisons": [
            {
                "rank": r["rank"],
                "ticker": r["ticker"],
                "trailing_return_pct": r["trailing_return_pct"],
                **comparison_by_ticker.get(r["ticker"], {}),
            }
            for r in rows
        ],
    }


@router.post("/publish-now", dependencies=[Depends(require_admin)])
async def publish_now(
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    force: bool = Query(False, description="Bypass the publish_signals_enabled gate for a controlled test."),
):
    """Manual trigger for the same publication the scheduler runs daily —
    for verifying the pipeline and catching up a missed day, not routine use.
    Respects the same off-by-default gate as the scheduled job unless
    force=true is passed explicitly, so a routine test call can't
    accidentally become the record's real first publication."""
    if not force and not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        raise HTTPException(
            409,
            "Publishing is currently disabled (publish_signals_enabled=false). "
            "Enable it via /admin/settings, or pass force=true for a one-off test.",
        )
    published = await publish_daily_signals(universe_id=universe_id, lookback_days=lookback_days)
    return {"published": published}


@router.get("/outcomes")
@limiter.limit("60/minute")
async def list_signal_outcomes(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS),
):
    """
    TR-4: live performance to date — public, unauthenticated, same as
    /published. Aggregate metrics (hit rate, information coefficient,
    quintile spread) plus the full per-date history, computed only from
    signal_outcomes (never blended with anything backtest-derived, per
    TR-4's hard requirement).
    """
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT target_date, ticker, rank, entry_price, exit_price,
                   realized_return_pct, benchmark_return_pct, beat_benchmark
            FROM signal_outcomes
            WHERE universe_id = $1 AND lookback_days = $2 AND horizon_days = $3
            ORDER BY target_date ASC, rank ASC
            """,
            universe_id, lookback_days, horizon_days,
        )

    row_dicts = [_record_to_dict(r) for r in rows]
    metrics = compute_outcome_metrics(row_dicts)

    return {
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "horizon_days": horizon_days,
        **metrics,
        "outcomes": [
            {**r, "target_date": str(r["target_date"])} for r in row_dicts
        ],
    }


@router.post("/evaluate-now", dependencies=[Depends(require_admin)])
async def evaluate_now(
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS),
):
    """Manual trigger for the same outcome evaluation the scheduler runs
    daily — for verifying the pipeline and catching up, not routine use.
    Unlike publish-now, no enable-gate: evaluating already-published,
    already-public history isn't itself a new disclosure."""
    evaluated = await evaluate_due_signal_outcomes(
        universe_id=universe_id, lookback_days=lookback_days, horizon_days=horizon_days
    )
    return {"evaluated": evaluated}
