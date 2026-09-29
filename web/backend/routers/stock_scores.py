"""
Phase 1 ("Trust") two-score system read endpoints (docs/stock-analysis-
requirements.html, SCR-1..4 / EXP-1..3). Public, unauthenticated by
design -- same posture as GET /signals/published, since these are
already-computed, non-personal facts about a stock, not a user's own
data.
"""

import json
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query, Request

from services.factor_narrative_service import (
    growth_sentence,
    low_vol_sentence,
    momentum_sentence,
    reversal_sentence,
    value_sentence,
)
from services.stock_score_capture_service import MOMENTUM_LOOKBACK_DAYS
from services.stock_score_service import flag_12week_trend, select_top_and_bottom_factors, weekly_change_explanation
from web.backend.db import service_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/stock-scores", tags=["stock-scores"])

DEFAULT_UNIVERSE = "All"

# EXP-4's drillable factors -- the same 5 keys factor_detail is always
# keyed by (see stock_score_capture_service.compute_and_persist_daily_scores).
FACTOR_KEYS = {"momentum", "reversal", "value", "growth", "low_vol"}


def _parse_factor_detail(row) -> dict:
    detail = row["factor_detail"]
    return json.loads(detail) if isinstance(detail, str) else detail


def _build_sentences(detail: dict) -> dict:
    momentum = detail.get("momentum", {})
    reversal = detail.get("reversal", {})
    value = detail.get("value", {})
    growth = detail.get("growth", {})
    low_vol = detail.get("low_vol", {})
    return {
        "momentum": momentum_sentence(momentum.get("raw"), MOMENTUM_LOOKBACK_DAYS, momentum.get("percentile")),
        "reversal": reversal_sentence(reversal.get("raw"), reversal.get("percentile")),
        "value": value_sentence(value.get("raw"), value.get("percentile")),
        "growth": growth_sentence(growth.get("raw_revenue"), growth.get("raw_earnings"), growth.get("percentile")),
        "low_vol": low_vol_sentence(low_vol.get("raw"), low_vol.get("percentile")),
    }


def _build_drivers_and_drags(detail: dict) -> dict:
    contributions = [
        {"factor": factor, "contribution": values["contribution"]}
        for factor, values in detail.items()
        if values.get("contribution") is not None
    ]
    return select_top_and_bottom_factors(contributions)


async def _fetch_latest_row(ticker: str, universe_id: str):
    async with service_conn() as conn:
        return await conn.fetchrow(
            """
            SELECT * FROM stock_scores
            WHERE ticker = $1 AND universe_id = $2
            ORDER BY as_of_date DESC LIMIT 1
            """,
            ticker.upper(), universe_id,
        )


@router.get("/{ticker}")
@limiter.limit("60/minute")
async def get_stock_score(request: Request, ticker: str, universe_id: str = Query(DEFAULT_UNIVERSE)):
    row = await _fetch_latest_row(ticker, universe_id)
    if row is None:
        raise HTTPException(404, f"No score on record yet for {ticker.upper()}.")

    detail = _parse_factor_detail(row)
    return {
        "ticker": row["ticker"],
        "as_of_date": row["as_of_date"].isoformat(),
        "universe_id": row["universe_id"],
        "short_score": row["short_score"],
        "short_signal": row["short_signal"],
        "short_confidence": {"score": row["short_confidence_score"], "label": row["short_confidence_label"]},
        "long_score": row["long_score"],
        "long_signal": row["long_signal"],
        "long_confidence": {"score": row["long_confidence_score"], "label": row["long_confidence_label"]},
        "sector_key": row["sector_key"],
        "short_sector_percentile": row["short_sector_percentile"],
        "long_sector_percentile": row["long_sector_percentile"],
        "factor_detail": detail,
        "explanations": _build_drivers_and_drags(detail),
        "sentences": _build_sentences(detail),
    }


@router.get("/{ticker}/history")
@limiter.limit("60/minute")
async def get_stock_score_history(
    request: Request, ticker: str, weeks: int = Query(12, ge=1, le=52), universe_id: str = Query(DEFAULT_UNIVERSE)
):
    since = date.today() - timedelta(weeks=weeks + 1)
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT as_of_date, short_score, long_score FROM stock_scores
            WHERE ticker = $1 AND universe_id = $2 AND as_of_date >= $3
            ORDER BY as_of_date
            """,
            ticker.upper(), universe_id, since,
        )
    short_history = [(r["as_of_date"], r["short_score"]) for r in rows if r["short_score"] is not None]
    long_history = [(r["as_of_date"], r["long_score"]) for r in rows if r["long_score"] is not None]
    return {
        "ticker": ticker.upper(),
        "short_term": flag_12week_trend(short_history, weeks=weeks),
        "long_term": flag_12week_trend(long_history, weeks=weeks),
    }


@router.get("/{ticker}/weekly-change")
@limiter.limit("60/minute")
async def get_stock_score_weekly_change(request: Request, ticker: str, universe_id: str = Query(DEFAULT_UNIVERSE)):
    today_row = await _fetch_latest_row(ticker, universe_id)
    if today_row is None:
        raise HTTPException(404, f"No score on record yet for {ticker.upper()}.")

    target = today_row["as_of_date"] - timedelta(days=7)
    async with service_conn() as conn:
        week_ago_row = await conn.fetchrow(
            """
            SELECT as_of_date, factor_detail FROM stock_scores
            WHERE ticker = $1 AND universe_id = $2 AND as_of_date <= $3
            ORDER BY as_of_date DESC LIMIT 1
            """,
            ticker.upper(), universe_id, target,
        )

    today_detail = _parse_factor_detail(today_row)
    week_ago_detail = _parse_factor_detail(week_ago_row) if week_ago_row is not None else None

    return {
        "ticker": ticker.upper(),
        "as_of_date": today_row["as_of_date"].isoformat(),
        "compared_to": week_ago_row["as_of_date"].isoformat() if week_ago_row is not None else None,
        "change": weekly_change_explanation(today_detail, week_ago_detail),
    }


@router.get("/{ticker}/factor/{factor}")
@limiter.limit("60/minute")
async def get_stock_factor_history(
    request: Request, ticker: str, factor: str, universe_id: str = Query(DEFAULT_UNIVERSE)
):
    """EXP-4: this ticker's own history for one factor (up to the last
    252 trading days on record -- naturally shorter than "1 year" until
    stock_scores itself has accumulated that much history; never padded
    to look longer than it is) alongside that same factor's sector
    median on each of those same days, so a user can see not just the
    trend but where the stock sits versus its peers.

    One query: ticker_history pulls this ticker's own raw/percentile per
    day, sector_medians aggregates every other stock_scores row sharing
    both that day and this ticker's sector_key into a single median --
    scoped to exactly the (date, sector) pairs ticker_history touches,
    not a full-table scan."""
    if factor not in FACTOR_KEYS:
        raise HTTPException(400, f"factor must be one of {sorted(FACTOR_KEYS)}")
    ticker = ticker.upper()

    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            WITH ticker_history AS (
                SELECT as_of_date, sector_key,
                       (factor_detail->$3->>'raw')::float8 AS raw,
                       (factor_detail->$3->>'percentile')::float8 AS percentile
                FROM stock_scores
                WHERE ticker = $1 AND universe_id = $2
                ORDER BY as_of_date DESC
                LIMIT 252
            ),
            sector_medians AS (
                SELECT s.as_of_date,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY (s.factor_detail->$3->>'raw')::float8) AS sector_median
                FROM stock_scores s
                JOIN (SELECT DISTINCT as_of_date, sector_key FROM ticker_history) th
                  ON s.as_of_date = th.as_of_date AND s.sector_key = th.sector_key
                WHERE s.universe_id = $2 AND (s.factor_detail->$3->>'raw') IS NOT NULL
                GROUP BY s.as_of_date
            )
            SELECT th.as_of_date, th.sector_key, th.raw, th.percentile, sm.sector_median
            FROM ticker_history th
            LEFT JOIN sector_medians sm ON sm.as_of_date = th.as_of_date
            ORDER BY th.as_of_date ASC
            """,
            ticker, universe_id, factor,
        )

    if not rows:
        raise HTTPException(404, f"No {factor} history on record yet for {ticker}.")

    return {
        "ticker": ticker,
        "factor": factor,
        "sector_key": rows[-1]["sector_key"],
        "history": [
            {
                "as_of_date": r["as_of_date"].isoformat(),
                "raw": r["raw"],
                "percentile": r["percentile"],
                "sector_median": r["sector_median"],
            }
            for r in rows
        ],
    }
