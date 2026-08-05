import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.pit_fundamentals_service import DEFAULT_UNIVERSE as FUNDAMENTALS_DEFAULT_UNIVERSE
from services.pit_fundamentals_service import capture_universe_fundamentals
from services.pit_price_service import DEFAULT_UNIVERSE, capture_universe_closes
from services.pit_universe_service import capture_universe_membership
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


async def capture_and_persist_universe_membership() -> int:
    """
    TR-3 Phase 2: snapshots today's INDEX_MAP/INDEX_FUND_UNIVERSE membership
    and inserts it into the append-only pit_universe_membership store. Same
    ON CONFLICT DO NOTHING immutability guarantee as pit_prices. Returns the
    number of rows actually inserted.
    """
    rows = await run_in_threadpool(capture_universe_membership)
    if not rows:
        return 0

    snapshot_date = date.today()
    inserted = 0
    async with service_conn() as conn:
        for row in rows:
            result = await conn.execute(
                """
                INSERT INTO pit_universe_membership (asset_type, universe_key, ticker, snapshot_date)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (asset_type, universe_key, ticker, snapshot_date) DO NOTHING
                """,
                row["asset_type"], row["universe_key"], row["ticker"], snapshot_date,
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info(
        "PIT universe membership capture: %d/%d rows newly inserted (rest already on record for today)",
        inserted, len(rows),
    )
    return inserted


async def capture_and_persist_fundamentals(universe_id: str = FUNDAMENTALS_DEFAULT_UNIVERSE) -> int:
    """
    TR-3 Phase 3: fetches today's forward_pe/revenue_growth/earnings_growth
    for each ticker in the universe and inserts it into the append-only
    pit_fundamentals store. Same ON CONFLICT DO NOTHING immutability
    guarantee as pit_prices. Returns the number of rows actually inserted.
    """
    rows = await run_in_threadpool(capture_universe_fundamentals, universe_id)
    if not rows:
        return 0

    as_of_date = date.today()
    inserted = 0
    async with service_conn() as conn:
        for row in rows:
            result = await conn.execute(
                """
                INSERT INTO pit_fundamentals (
                    ticker, as_of_date, forward_pe, revenue_growth_pct, earnings_growth_pct, source
                ) VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (ticker, as_of_date) DO NOTHING
                """,
                row["ticker"], as_of_date, row["forward_pe"], row["revenue_growth_pct"],
                row["earnings_growth_pct"], row["source"],
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info(
        "PIT fundamentals capture for %s: %d/%d tickers newly inserted (rest already on record for today)",
        universe_id, inserted, len(rows),
    )
    return inserted
