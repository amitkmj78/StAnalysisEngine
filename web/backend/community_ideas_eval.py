"""COM-2: the DB/yfinance-aware orchestration for scoring published
community ideas at their horizon -- services/community_idea_service.py
stays pure (no DB, no network), same layering convention
web/backend/condition_alerts_eval.py already uses for services/
condition_alert_service.py.
"""

import logging

from starlette.concurrency import run_in_threadpool

from services.community_idea_service import evaluate_idea_outcome
from services.signal_publication_service import compute_spy_returns_for_dates, fetch_spy_close_series
from services.yfinance_cache import get_cached_history
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

PRICE_HISTORY_PERIOD = "2y"


def _fetch_closes(tickers: list[str]) -> dict:
    out = {}
    for ticker in tickers:
        hist = get_cached_history(ticker, PRICE_HISTORY_PERIOD, auto_adjust=True)
        if not hist.empty:
            out[ticker] = hist["Close"]
    return out


async def evaluate_due_community_ideas() -> int:
    """Checks every unscored idea against its own ticker's close
    series. An idea whose horizon hasn't elapsed yet is silently left
    unscored (evaluate_idea_outcome returns None) -- never guessed at,
    picked up again next run. Scored exactly once: the row's own
    scored_at IS NULL filter means a later run never re-touches it."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM community_ideas WHERE scored_at IS NULL")
    if not rows:
        return 0

    tickers = sorted({r["ticker"] for r in rows})
    closes_by_ticker, spy_close = await run_in_threadpool(
        lambda: (_fetch_closes(tickers), fetch_spy_close_series())
    )

    scored = 0
    async with service_conn() as conn:
        for row in rows:
            closes = closes_by_ticker.get(row["ticker"])
            if closes is None:
                continue
            as_of = row["created_at"].date()
            outcome = await run_in_threadpool(
                evaluate_idea_outcome, as_of, row["direction"], closes, row["horizon_days"]
            )
            if outcome is None:
                continue
            spy_return_by_date = compute_spy_returns_for_dates(spy_close, [as_of], row["horizon_days"])
            spy_return = spy_return_by_date.get(as_of)
            if spy_return is None:
                continue
            excess = round(outcome["realized_return_pct"] - spy_return, 4)
            await conn.execute(
                """
                UPDATE community_ideas
                SET realized_return_pct = $2, excess_vs_spy_pct = $3, outcome = $4, scored_at = now()
                WHERE id = $1
                """,
                row["id"], outcome["realized_return_pct"], excess, outcome["outcome"],
            )
            scored += 1

    if scored:
        logger.info("Scored %d community ideas", scored)
    return scored
