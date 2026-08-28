import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from starlette.concurrency import run_in_threadpool

from services.pit_analyst_rating_service import DEFAULT_UNIVERSE as ANALYST_RATING_DEFAULT_UNIVERSE
from services.pit_analyst_rating_service import capture_universe_analyst_ratings
from services.pit_fundamentals_service import DEFAULT_UNIVERSE as FUNDAMENTALS_DEFAULT_UNIVERSE
from services.pit_fundamentals_service import capture_universe_fundamentals
from services.pit_price_service import DEFAULT_UNIVERSE, capture_universe_closes
from services.pit_quant_signal_service import DEFAULT_UNIVERSE as QUANT_SIGNAL_DEFAULT_UNIVERSE
from services.pit_quant_signal_service import capture_universe_quant_signals
from services.pit_universe_service import capture_universe_membership
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

_EASTERN = ZoneInfo("America/New_York")


def _eastern_today() -> date:
    """The server runs on UTC system time, so a naive date.today() call made
    in the evening (after 8pm ET / midnight UTC) mislabels data with
    tomorrow's date. Ratings/fundamentals/signals should be stamped with the
    US trading day they were captured for, not the server's UTC calendar
    day."""
    return datetime.now(_EASTERN).date()


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

    snapshot_date = _eastern_today()
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

    as_of_date = _eastern_today()
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


async def capture_and_persist_quant_signals(universe_id: str = QUANT_SIGNAL_DEFAULT_UNIVERSE) -> int:
    """
    Fetches today's Quant Signal (BUY/HOLD/SELL, same as /predict and the
    Stock Screener) for each ticker in the universe and inserts it into
    the append-only pit_quant_signal store. Same ON CONFLICT DO NOTHING
    immutability guarantee as pit_prices. CPU-heavy (trains a model per
    ticker) — meant to run off-hours, see scheduler.py. Returns the
    number of rows actually inserted.
    """
    rows = await run_in_threadpool(capture_universe_quant_signals, universe_id)
    if not rows:
        return 0

    as_of_date = _eastern_today()
    inserted = 0
    async with service_conn() as conn:
        for row in rows:
            result = await conn.execute(
                """
                INSERT INTO pit_quant_signal (
                    ticker, as_of_date, signal, expected_return_pct, target_price, last_close, source
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker, as_of_date) DO NOTHING
                """,
                row["ticker"], as_of_date, row["signal"], row["expected_return_pct"],
                row["target_price"], row["last_close"], row["source"],
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info(
        "PIT quant signal capture for %s: %d/%d tickers newly inserted (rest already on record for today)",
        universe_id, inserted, len(rows),
    )
    return inserted


async def capture_and_persist_analyst_ratings(universe_id: str = ANALYST_RATING_DEFAULT_UNIVERSE) -> int:
    """
    Fetches today's real, third-party analyst consensus (same as the
    Stock Screener's "Analyst Rating" column) for each ticker with
    coverage in the universe and inserts it into the append-only
    pit_analyst_rating store. Same ON CONFLICT DO NOTHING immutability
    guarantee as pit_prices. Returns the number of rows actually inserted.
    """
    rows = await run_in_threadpool(capture_universe_analyst_ratings, universe_id)
    if not rows:
        return 0

    as_of_date = _eastern_today()
    inserted = 0
    async with service_conn() as conn:
        for row in rows:
            result = await conn.execute(
                """
                INSERT INTO pit_analyst_rating (
                    ticker, as_of_date, consensus, analyst_count, buy_pct,
                    target_mean, target_high, target_low, current_price, source
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (ticker, as_of_date) DO NOTHING
                """,
                row["ticker"], as_of_date, row["consensus"], row["analyst_count"], row["buy_pct"],
                row["target_mean"], row["target_high"], row["target_low"], row["current_price"], row["source"],
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info(
        "PIT analyst rating capture for %s: %d/%d tickers newly inserted (rest already on record for today)",
        universe_id, inserted, len(rows),
    )
    return inserted
