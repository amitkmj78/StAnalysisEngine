import logging
from datetime import date, datetime, timedelta, timezone

from starlette.concurrency import run_in_threadpool

from services.signal_publication_service import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TOP_N,
    DEFAULT_UNIVERSE,
    build_daily_signal_set_hybrid,
    evaluate_signal_outcomes_for_date,
    get_model_version_hash,
)
from web.backend.db import service_conn
from web.backend.pit_signals import fetch_pit_prices_as_of

logger = logging.getLogger(__name__)


async def is_publication_recorded(
    target_date: date,
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> bool:
    """NFR-01/02: the same existence check publish_daily_signals uses to
    stay idempotent, extracted so the alert jobs can ask "did today's
    publication actually happen?" without duplicating the query."""
    async with service_conn() as conn:
        count = await conn.fetchval(
            """
            SELECT count(*) FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            """,
            target_date, universe_id, lookback_days,
        )
    return bool(count)


async def publish_daily_signals(
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_N,
    target_date: date | None = None,
) -> int:
    """
    TR-1: commits today's Signal Set to the public, append-only ledger.
    Skips (returns 0) if this target_date/universe/lookback combination
    already has a non-corrected publication — makes this safely re-runnable
    from a scheduler restart or a manual admin trigger without ever
    double-publishing or overwriting what's already public. Returns the
    number of rows published.
    """
    target_date = target_date or date.today()

    async with service_conn() as conn:
        existing = await conn.fetchval(
            """
            SELECT count(*) FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3
              AND reason_code IS NULL
            """,
            target_date, universe_id, lookback_days,
        )
        if existing:
            logger.info(
                "Publication already exists for %s/%s/%dd — skipping",
                target_date, universe_id, lookback_days,
            )
            return 0

        pit_price_rows = await fetch_pit_prices_as_of(target_date, universe_id, lookback_days)
        rows = await run_in_threadpool(
            build_daily_signal_set_hybrid, pit_price_rows, universe_id, lookback_days, top_n
        )
        if not rows:
            logger.warning(
                "No signal rows computed for %s/%s/%dd — nothing published",
                target_date, universe_id, lookback_days,
            )
            return 0

        model_hash = get_model_version_hash()
        as_of = datetime.now(timezone.utc)
        for row in rows:
            await conn.execute(
                """
                INSERT INTO published_signals (
                    model_version_hash, as_of_data_timestamp, target_date, universe_id,
                    lookback_days, rank, ticker, trailing_return_pct, data_source
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                model_hash, as_of, target_date, universe_id, lookback_days,
                row["rank"], row["ticker"], row["trailing_return_pct"], row["data_source"],
            )

    pit_sourced = sum(1 for r in rows if r["data_source"] == "pit")
    logger.info(
        "Published %d signals for %s/%s/%dd (model %s) — %d/%d from PIT store",
        len(rows), target_date, universe_id, lookback_days, model_hash[:12], pit_sourced, len(rows),
    )
    return len(rows)


async def evaluate_due_signal_outcomes(
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> int:
    """
    TR-4: finds every published target_date old enough that horizon_days
    trading days have plausibly elapsed (a loose calendar-day prefilter —
    evaluate_signal_outcomes_for_date does the real trading-day check and
    returns None if it's not actually due yet) and doesn't already have
    outcomes recorded, computes the realized outcome, and stores it.
    Idempotent via signal_outcomes' unique constraint. Returns the number
    of outcome rows inserted.
    """
    # ~1.6x calendar/trading-day ratio (weekends + holidays) plus a small
    # buffer — deliberately loose since the real gate is the trading-day
    # check inside evaluate_signal_outcomes_for_date, not this prefilter.
    cutoff = date.today() - timedelta(days=int(horizon_days * 1.6) + 5)

    async with service_conn() as conn:
        due_dates = await conn.fetch(
            """
            SELECT DISTINCT ps.target_date FROM published_signals ps
            WHERE ps.universe_id = $1 AND ps.lookback_days = $2 AND ps.reason_code IS NULL
              AND ps.target_date <= $3
              AND NOT EXISTS (
                SELECT 1 FROM signal_outcomes so
                WHERE so.target_date = ps.target_date AND so.universe_id = ps.universe_id
                  AND so.lookback_days = ps.lookback_days AND so.horizon_days = $4
              )
            ORDER BY ps.target_date
            """,
            universe_id, lookback_days, cutoff, horizon_days,
        )

        total_inserted = 0
        for d in due_dates:
            target_date = d["target_date"]
            rows = await conn.fetch(
                """
                SELECT ticker, rank FROM published_signals
                WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
                """,
                target_date, universe_id, lookback_days,
            )
            row_dicts = [{"ticker": r["ticker"], "rank": r["rank"]} for r in rows]

            outcomes = await run_in_threadpool(
                evaluate_signal_outcomes_for_date, row_dicts, target_date, universe_id, horizon_days
            )
            if not outcomes:
                continue  # not actually due yet, or no usable price data

            for o in outcomes:
                await conn.execute(
                    """
                    INSERT INTO signal_outcomes (
                        target_date, universe_id, lookback_days, horizon_days, ticker, rank,
                        entry_price, exit_price, realized_return_pct, benchmark_return_pct, beat_benchmark
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (target_date, universe_id, lookback_days, horizon_days, ticker) DO NOTHING
                    """,
                    target_date, universe_id, lookback_days, horizon_days, o["ticker"], o["rank"],
                    o["entry_price"], o["exit_price"], o["realized_return_pct"],
                    o["benchmark_return_pct"], o["beat_benchmark"],
                )
                total_inserted += 1

    if total_inserted:
        logger.info("Evaluated %d signal outcomes across %d due dates", total_inserted, len(due_dates))
    return total_inserted
