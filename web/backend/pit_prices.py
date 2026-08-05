import logging

from starlette.concurrency import run_in_threadpool

from services.pit_price_service import DEFAULT_UNIVERSE, capture_universe_closes
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


async def capture_and_persist_pit_prices(universe_id: str = DEFAULT_UNIVERSE) -> int:
    """
    TR-3 Phase 1: fetches today's close for each ticker in the universe and
    inserts it into the append-only pit_prices store. ON CONFLICT DO NOTHING
    on (ticker, price_date) makes this safely re-runnable (scheduler
    restart, manual admin trigger, catching up after downtime) without ever
    overwriting a row already on record — the point-in-time guarantee comes
    from a row never changing once captured. Returns the number of rows
    actually inserted (new price_dates only, not re-runs of an already-
    captured day).
    """
    rows = await run_in_threadpool(capture_universe_closes, universe_id)
    if not rows:
        return 0

    inserted = 0
    async with service_conn() as conn:
        for row in rows:
            result = await conn.execute(
                """
                INSERT INTO pit_prices (ticker, price_date, close, source)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (ticker, price_date) DO NOTHING
                """,
                row["ticker"], row["price_date"], row["close"], row["source"],
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info(
        "PIT capture for %s: %d/%d tickers newly inserted (rest already on record for today)",
        universe_id, inserted, len(rows),
    )
    return inserted
