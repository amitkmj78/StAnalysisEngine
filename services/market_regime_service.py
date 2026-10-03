"""
Orchestrates the (gate-failed, explicitly-overridden — see
services/market_internals_service.py's module docstring) regime-scoring
engine into a persisted daily table, and reads it back for the live app.
Unlike market_internals_service.py this module is NOT pure: it fetches
live data and talks to Postgres via web.backend.db.service_conn, the
same cross-user/service-role pattern services/stock_score_capture_service.py
already uses for its own daily capture-and-persist job.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
from starlette.concurrency import run_in_threadpool

from services.market_data_service import fetch_dimension_frame, fetch_market_internals_history
from services.regime_dimensions import regime_dimensions
from services.market_internals_service import (
    apply_hysteresis,
    compute_composite_score,
    compute_internals_components,
    compute_internals_score,
    map_regime,
)
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

# Shown permanently beside the regime banner. Plain language (REG-12); the
# statistics and test windows are on the methodology page below. This line must
# keep saying the signal is not validated and was shown anyway -- see
# market_internals_service.py's module docstring.
REGIME_GATE_DISCLOSURE = (
    "This signal has not been validated. Its own tests failed three times, so it is "
    "shown for information only, not as a recommendation. How it was tested is on the "
    "methodology page."
)

# The full statistics behind the disclosure, for the methodology page (REG-12).
# The -7.91% / p<0.0001 figures come from the spec's §9c backtest (Attempt 3).
REGIME_METHODOLOGY = [
    "The regime signal failed its own release gate three times in backtesting. In its most recent "
    "test, isolating the 2008 financial crisis (2007-06 through 2009-06), extreme internals stress "
    "predicted a further -7.91% over the next 21 days (n=24, p<0.0001 after correcting for "
    "overlapping-window autocorrelation). That is the opposite of a 'buy the fear' signal, and the "
    "Risk-On/Risk-Off framing would have compounded losses rather than flagged an opportunity. "
    "This is a condition label, not a recommendation.",
    "Forward-risk test (REG-10), run once on the stored labels and market history, with the test "
    "agreed before the run and no thresholds changed afterwards. Label test: insufficient data -- "
    "the history has 25 Cautious days and no Risk-On days, so the comparison cannot be made. "
    "Divergence test: insufficient data -- 7 flagged days against 526 unflagged. Of the groups that "
    "could be compared, Cautious days showed higher average forward 21-day volatility (15.95%) than "
    "Neutral (12.03%) and Constructive (10.53%). Forward windows overlap, so days are not "
    "independent, and no significance is claimed.",
    "Breadth history (REG-11) is built from the current S&P 500 universe. Stocks that have since left "
    "the index are not included, so breadth history is subject to survivorship bias.",
    "The rates, credit, and breadth readings beside the banner are display-only. They do not change "
    "the regime label or the trading agent's caps.",
]

_PERSIST_COLUMNS = [
    "as_of_date", "internals_score", "mds", "regime_raw", "regime_confirmed",
    "data_completeness", "conflict_flag",
    "breadth_50dma", "vix", "vix3m", "xly_xlp", "hyg_ief", "rsp_spy",
]


def _compute_regime_frame(internals_history: pd.DataFrame) -> pd.DataFrame:
    """Chains the gated scoring functions over every date in
    `internals_history` to produce one row per scoreable date. Dates
    still inside the 250-day z-score warm-up window (compute_internals_score
    returns NaN there) are skipped, not written as a bogus row — a raw
    None can't be a hysteresis candidate either. Hysteresis is applied
    once, across the whole raw-label series, so a backfill's early
    "confirmed" regimes are exactly what the daily job would have
    produced on those historical days, not artifacts of only ever seeing
    one day at a time."""
    internals_scores = compute_internals_score(internals_history)

    records = []
    for as_of, score in internals_scores.items():
        if pd.isna(score):
            continue
        row = internals_history.loc[as_of]
        composite = compute_composite_score(internals_score=float(score))
        records.append(
            {
                "as_of_date": as_of,
                "internals_score": float(score),
                "mds": composite["mds"],
                "data_completeness": composite["data_completeness"],
                "conflict_flag": composite["conflict_flag"],
                "regime_raw": map_regime(composite["mds"]),
                "breadth_50dma": float(row["breadth_50dma"]),
                "vix": float(row["vix"]),
                "vix3m": float(row["vix3m"]),
                "xly_xlp": float(row["xly_xlp"]),
                "hyg_ief": float(row["hyg_ief"]),
                "rsp_spy": float(row["rsp_spy"]),
            }
        )

    if not records:
        return pd.DataFrame(columns=_PERSIST_COLUMNS)

    frame = pd.DataFrame(records).set_index("as_of_date")
    frame["regime_confirmed"] = apply_hysteresis(frame["regime_raw"])
    return frame.reset_index()[_PERSIST_COLUMNS]


async def _persist_regime_rows(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    params = [
        (
            r["as_of_date"].date() if hasattr(r["as_of_date"], "date") else r["as_of_date"],
            r["internals_score"], r["mds"], r["regime_raw"], r["regime_confirmed"],
            r["data_completeness"], bool(r["conflict_flag"]),
            r["breadth_50dma"], r["vix"], r["vix3m"], r["xly_xlp"], r["hyg_ief"], r["rsp_spy"],
        )
        for _, r in rows.iterrows()
    ]
    async with service_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO market_regime_daily (
                as_of_date, internals_score, mds, regime_raw, regime_confirmed,
                data_completeness, conflict_flag,
                breadth_50dma, vix, vix3m, xly_xlp, hyg_ief, rsp_spy
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (as_of_date) DO UPDATE SET
                internals_score = $2, mds = $3, regime_raw = $4, regime_confirmed = $5,
                data_completeness = $6, conflict_flag = $7,
                breadth_50dma = $8, vix = $9, vix3m = $10, xly_xlp = $11, hyg_ief = $12, rsp_spy = $13,
                computed_at_utc = now()
            """,
            params,
        )
    return len(params)


async def compute_and_persist_daily_regime(backfill: bool = False) -> dict:
    """The function both the daily scheduler job (backfill=False, latest
    row only) and the one-time admin backfill endpoint (backfill=True,
    every scoreable date in the fetched window) call. Never triggered
    automatically in backfill mode — see the admin endpoint's own
    docstring for why."""
    history = await run_in_threadpool(fetch_market_internals_history, "3y")
    if history.empty:
        logger.warning("Market regime: fetch_market_internals_history returned no data, nothing to persist")
        return {"rows_persisted": 0}

    frame = _compute_regime_frame(history)
    if frame.empty:
        logger.warning("Market regime: no scoreable rows yet (250-day warm-up window not full)")
        return {"rows_persisted": 0}

    rows_to_persist = frame if backfill else frame.tail(1)
    persisted = await _persist_regime_rows(rows_to_persist)
    latest = frame.iloc[-1]
    return {
        "rows_persisted": persisted,
        "latest_as_of_date": latest["as_of_date"].date().isoformat(),
        "latest_regime": latest["regime_confirmed"],
    }


async def regime_as_of(target_date: Optional[date] = None) -> Optional[str]:
    """The correct regime for the context being rendered — today's for a
    live holding, the historical as-of-that-date regime for a track-record
    row. Deliberately does NOT default to "latest" when a historical date
    should be used instead; callers rendering historical rows must pass
    their own target_date rather than relying on this function's default."""
    async with service_conn() as conn:
        if target_date is None:
            row = await conn.fetchrow(
                "SELECT regime_confirmed FROM market_regime_daily ORDER BY as_of_date DESC LIMIT 1"
            )
        else:
            row = await conn.fetchrow(
                "SELECT regime_confirmed FROM market_regime_daily WHERE as_of_date = $1", target_date
            )
    return row["regime_confirmed"] if row else None


async def get_regime_snapshot() -> Optional[dict]:
    """Everything GET /api/v1/market/regime needs in one call: the latest
    persisted row plus a fresh compute_internals_components() reading
    (components aren't persisted per-row, so these are recomputed from
    the same 15-min-cached history fetch the daily job itself used).
    Returns None if market_regime_daily has no rows yet (e.g. before the
    first backfill/scheduler run) — the caller decides how to render
    that, not this function."""
    async with service_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT as_of_date, regime_confirmed, regime_raw, mds, internals_score,
                   data_completeness, conflict_flag
            FROM market_regime_daily ORDER BY as_of_date DESC LIMIT 1
            """
        )
    if row is None:
        return None

    history = await run_in_threadpool(fetch_market_internals_history, "3y")
    components = compute_internals_components(history) if not history.empty else {}
    dimension_history = await run_in_threadpool(fetch_dimension_frame, "3y")
    dimensions = regime_dimensions(dimension_history) if not dimension_history.empty else {}

    return {
        "as_of_date": row["as_of_date"].isoformat(),
        "regime": row["regime_confirmed"],
        "regime_raw": row["regime_raw"],
        "mds": row["mds"],
        "internals_score": row["internals_score"],
        "data_completeness": row["data_completeness"],
        "conflict_flag": row["conflict_flag"],
        "components": components,
        "dimensions": dimensions,
        "disclosure": REGIME_GATE_DISCLOSURE,
        "methodology": REGIME_METHODOLOGY,
    }
