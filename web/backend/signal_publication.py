import logging
from datetime import date, datetime, timezone

from starlette.concurrency import run_in_threadpool

from services.signal_publication_service import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TOP_N,
    DEFAULT_UNIVERSE,
    build_daily_signal_set,
    get_model_version_hash,
)
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


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

        rows = await run_in_threadpool(build_daily_signal_set, universe_id, lookback_days, top_n)
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
                    lookback_days, rank, ticker, trailing_return_pct
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                model_hash, as_of, target_date, universe_id, lookback_days,
                row["rank"], row["ticker"], row["trailing_return_pct"],
            )

    logger.info(
        "Published %d signals for %s/%s/%dd (model %s)",
        len(rows), target_date, universe_id, lookback_days, model_hash[:12],
    )
    return len(rows)
