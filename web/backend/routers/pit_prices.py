from datetime import date

from fastapi import APIRouter, Depends, Query

from services.pit_fundamentals_service import DEFAULT_UNIVERSE as FUNDAMENTALS_DEFAULT_UNIVERSE
from services.pit_price_service import DEFAULT_UNIVERSE
from web.backend.admin import require_admin
from web.backend.db import service_conn
from web.backend.pit_prices import (
    capture_and_persist_fundamentals,
    capture_and_persist_pit_prices,
    capture_and_persist_universe_membership,
)
from web.backend.pit_signals import reconcile_published_vs_pit

router = APIRouter(
    prefix="/api/v1/pit-prices",
    tags=["pit-prices"],
    dependencies=[Depends(require_admin)],
)


@router.get("/status")
async def pit_prices_status(universe_id: str = Query(DEFAULT_UNIVERSE)):
    """
    TR-3 progress check: how much point-in-time history has actually
    accumulated across all three stores (prices, universe membership,
    fundamentals) yet. Nothing consumes this data yet, so this is the only
    way to see it's working before enough days have piled up to be useful
    for anything.
    """
    async with service_conn() as conn:
        prices_row = await conn.fetchrow(
            """
            SELECT count(*) AS row_count,
                   count(DISTINCT ticker) AS ticker_count,
                   count(DISTINCT price_date) AS days_captured,
                   min(price_date) AS earliest_date,
                   max(price_date) AS latest_date,
                   max(captured_at_utc) AS last_captured_at_utc
            FROM pit_prices
            """
        )
        membership_row = await conn.fetchrow(
            """
            SELECT count(*) AS row_count,
                   count(DISTINCT universe_key) AS universe_count,
                   count(DISTINCT snapshot_date) AS days_captured,
                   min(snapshot_date) AS earliest_date,
                   max(snapshot_date) AS latest_date,
                   max(captured_at_utc) AS last_captured_at_utc
            FROM pit_universe_membership
            """
        )
        fundamentals_row = await conn.fetchrow(
            """
            SELECT count(*) AS row_count,
                   count(DISTINCT ticker) AS ticker_count,
                   count(DISTINCT as_of_date) AS days_captured,
                   min(as_of_date) AS earliest_date,
                   max(as_of_date) AS latest_date,
                   max(captured_at_utc) AS last_captured_at_utc
            FROM pit_fundamentals
            """
        )
    return {
        "universe_id": universe_id,
        "prices": dict(prices_row),
        "universe_membership": dict(membership_row),
        "fundamentals": dict(fundamentals_row),
    }


@router.post("/capture-now")
async def capture_now(universe_id: str = Query(DEFAULT_UNIVERSE)):
    """Manual trigger for the same capture the scheduler runs daily (all
    three stores) — for verifying the pipeline and catching up a missed
    day, not routine use. No enable-gate: unlike publish-now, capturing
    this data is neither public nor irreversible, so there's no
    accidental-first-publication risk to guard against."""
    prices_inserted = await capture_and_persist_pit_prices(universe_id=universe_id)
    membership_inserted = await capture_and_persist_universe_membership()
    fundamentals_inserted = await capture_and_persist_fundamentals(
        universe_id=universe_id if universe_id else FUNDAMENTALS_DEFAULT_UNIVERSE
    )
    return {
        "prices_inserted": prices_inserted,
        "universe_membership_inserted": membership_inserted,
        "fundamentals_inserted": fundamentals_inserted,
    }


@router.get("/reconcile")
async def reconcile(
    target_date: date = Query(...),
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(30),
    top_n: int = Query(25),
):
    """
    TR-6's acceptance test, on demand: reconstructs target_date's momentum
    ranking purely from PIT data known as of that date, and reports
    whether it matches what was actually published. `byte_identical` is
    only true once the PIT store has enough trailing history for every
    published ticker — before then, `missing_from_pit_history` says
    exactly which tickers and why, rather than a false pass or a bare
    failure.
    """
    return await reconcile_published_vs_pit(
        target_date=target_date, universe_id=universe_id, lookback_days=lookback_days, top_n=top_n
    )
