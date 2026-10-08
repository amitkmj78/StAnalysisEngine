"""ALX-1: the DB/yfinance-aware orchestration for condition_alerts --
services/condition_alert_service.py stays pure (no DB, no network), same
layering convention web/backend/signal_publication.py already uses for
services/signal_publication_service.py.
"""

import json
import logging

from starlette.concurrency import run_in_threadpool

from services.condition_alert_service import (
    build_condition_frame,
    evaluate_condition_alert,
    is_intraday_eligible,
    parse_condition,
)
from services.notification_dispatcher import dispatch_alert
from services.strategy_engine import Rule
from services.yfinance_cache import get_cached_history, get_earnings_report_dates
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

# Enough trading days for the longest indicator a condition can reference
# (dist_52w_high_pct/close_vs_sma_200_pct need a full year, plus warm-up).
PRICE_HISTORY_PERIOD = "2y"

# ALX-2: "at most 1-minute delay with licensed real-time data, otherwise
# labeled as delayed" -- this app has no licensed real-time feed (see
# services/price_provider.py), so every intraday fire is explicitly
# labeled with that disclosure rather than implying real-time delivery.
INTRADAY_INTERVAL = "1m"
INTRADAY_PERIOD = "1d"
INTRADAY_LATENCY_DISCLOSURE = (
    "Based on intraday bars from this app's active price-data provider, "
    "which is not a licensed real-time feed -- actual latency depends on "
    "that provider and is not guaranteed to be under 1 minute."
)


async def _fetch_regime_by_date() -> dict[str, str]:
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT as_of_date, regime_confirmed FROM market_regime_daily")
    return {r["as_of_date"].isoformat(): r["regime_confirmed"] for r in rows}


async def _fetch_score_history(ticker: str) -> list[dict]:
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT as_of_date, short_score, long_score, short_signal, long_signal
            FROM stock_scores WHERE ticker = $1 AND universe_id = 'All' ORDER BY as_of_date
            """,
            ticker,
        )
    return [dict(r) for r in rows]


def _build_frame_for_ticker(ticker: str, regime_by_date: dict[str, str], score_history: list[dict]):
    hist = get_cached_history(ticker, PRICE_HISTORY_PERIOD, auto_adjust=True)
    if hist.empty:
        return None
    reports = get_earnings_report_dates(ticker)
    return build_condition_frame(hist, regime_by_date, reports, score_history)


async def evaluate_due_condition_alerts() -> int:
    """Checks every active, not-yet-triggered condition_alert's AND/OR
    combination against that ticker's latest price/indicator/score/
    signal/regime snapshot. One price-history + score-history fetch per
    DISTINCT ticker (not per alert), same "one bulk query/fetch per
    ticker, not N" idiom web/backend/scheduler.py's
    _evaluate_watchlist_alerts already uses for score_* conditions."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM condition_alerts WHERE active AND triggered_at IS NULL")
    if not rows:
        return 0

    regime_by_date = await _fetch_regime_by_date()
    tickers = sorted({r["ticker"] for r in rows})
    frames = {}
    for ticker in tickers:
        score_history = await _fetch_score_history(ticker)
        frame = await run_in_threadpool(_build_frame_for_ticker, ticker, regime_by_date, score_history)
        if frame is not None:
            frames[ticker] = frame

    triggered = 0
    async with service_conn() as conn:
        for row in rows:
            frame = frames.get(row["ticker"])
            if frame is None:
                continue
            fired = await _check_and_fire(conn, row, frame)
            triggered += 1 if fired else 0

    if triggered:
        logger.info("Evaluated condition alerts across %d tickers, triggered %d", len(frames), triggered)
    return triggered


def _parse_row_conditions(row) -> tuple[list[Rule] | None, list | None]:
    raw_conditions = row["conditions"]
    if isinstance(raw_conditions, str):
        raw_conditions = json.loads(raw_conditions)
    try:
        return [parse_condition(c) for c in raw_conditions], raw_conditions
    except ValueError as e:
        logger.warning("condition_alerts id=%s has an unparseable condition, skipping: %s", row["id"], e)
        return None, None


async def _check_and_fire(conn, row, frame, latency_note: str | None = None) -> bool:
    """Shared by both the daily and intraday jobs: parse -> evaluate ->
    persist + dispatch on a real fire. Returns whether it fired."""
    conditions, raw_conditions = _parse_row_conditions(row)
    if conditions is None:
        return False
    fired = evaluate_condition_alert(conditions, row["combinator"], frame)
    if fired is not True:
        return False

    latest = frame.iloc[-1]
    detail_parts = [f"{c.field} {c.op} {c.value}" for c in conditions]
    detail = f"{row['combinator']} of: " + ", ".join(detail_parts) + f" (price ${latest['price']:.2f})"
    if latency_note:
        detail = f"{detail}. {latency_note}"
    await conn.execute(
        "UPDATE condition_alerts SET triggered_at = now(), triggered_detail = $2 WHERE id = $1",
        row["id"], detail,
    )
    await dispatch_alert(
        str(row["user_id"]), row["ticker"], "condition_alert",
        f"{row['ticker']}: condition alert triggered",
        f"{row['ticker']} met your alert condition ({row['combinator']}): {detail}",
        {"combinator": row["combinator"], "conditions": raw_conditions, "price": float(latest["price"])},
    )
    return True


async def evaluate_due_intraday_condition_alerts() -> int:
    """ALX-2: the same check as evaluate_due_condition_alerts above, but
    on 1-minute intraday bars and restricted to alerts whose conditions
    are ENTIRELY price/indicator fields (services.condition_alert_
    service.is_intraday_eligible) -- score/signal/regime/earnings
    conditions are daily snapshots that this faster poll can't move the
    needle on, so they're left to the daily job. Every fire here is
    explicitly latency-disclosed (INTRADAY_LATENCY_DISCLOSURE), not
    presented as real-time. A long-window indicator (close_vs_sma_200_pct,
    dist_52w_high_pct) simply won't have enough bars within a single
    intraday ("1d") fetch to resolve -- evaluate_condition_alert already
    returns None (not guessed at) for that, no special-casing needed here."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM condition_alerts WHERE active AND triggered_at IS NULL")
    if not rows:
        return 0

    eligible_rows = []
    for row in rows:
        conditions, _ = _parse_row_conditions(row)
        if conditions is not None and is_intraday_eligible(conditions):
            eligible_rows.append(row)
    if not eligible_rows:
        return 0

    tickers = sorted({r["ticker"] for r in eligible_rows})

    def _fetch_intraday(ticker: str):
        hist = get_cached_history(ticker, INTRADAY_PERIOD, auto_adjust=True, interval=INTRADAY_INTERVAL)
        if hist.empty:
            return None
        return build_condition_frame(hist, regime_by_date=None, earnings_reports=[], score_history=None)

    frames = {}
    for ticker in tickers:
        frame = await run_in_threadpool(_fetch_intraday, ticker)
        if frame is not None:
            frames[ticker] = frame

    triggered = 0
    async with service_conn() as conn:
        for row in eligible_rows:
            frame = frames.get(row["ticker"])
            if frame is None:
                continue
            fired = await _check_and_fire(conn, row, frame, latency_note=INTRADAY_LATENCY_DISCLOSURE)
            triggered += 1 if fired else 0

    if triggered:
        logger.info("Evaluated intraday condition alerts across %d tickers, triggered %d", len(frames), triggered)
    return triggered
