from fastapi import APIRouter, Depends, Query

from services.pit_price_service import DEFAULT_UNIVERSE
from web.backend.admin import require_admin
from web.backend.db import service_conn
from web.backend.pit_prices import capture_and_persist_pit_prices

router = APIRouter(
    prefix="/api/v1/pit-prices",
    tags=["pit-prices"],
    dependencies=[Depends(require_admin)],
)


@router.get("/status")
async def pit_prices_status(universe_id: str = Query(DEFAULT_UNIVERSE)):
    """
    TR-3 Phase 1 progress check: how much point-in-time history has
    actually accumulated for this universe yet. Nothing consumes this data
    yet, so this is the only way to see it's working before enough days
    have piled up to be useful for anything.
    """
    async with service_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT count(*) AS row_count,
                   count(DISTINCT ticker) AS ticker_count,
                   count(DISTINCT price_date) AS trading_days_captured,
                   min(price_date) AS earliest_date,
                   max(price_date) AS latest_date,
                   max(captured_at_utc) AS last_captured_at_utc
            FROM pit_prices
            """
        )
    return {
        "universe_id": universe_id,
        "row_count": row["row_count"],
        "ticker_count": row["ticker_count"],
        "trading_days_captured": row["trading_days_captured"],
        "earliest_date": row["earliest_date"],
        "latest_date": row["latest_date"],
        "last_captured_at_utc": row["last_captured_at_utc"],
    }


@router.post("/capture-now")
async def capture_now(universe_id: str = Query(DEFAULT_UNIVERSE)):
    """Manual trigger for the same capture the scheduler runs daily —
    for verifying the pipeline and catching up a missed day, not routine
    use. No enable-gate: unlike publish-now, capturing a price is neither
    public nor irreversible, so there's no accidental-first-publication
    risk to guard against."""
    inserted = await capture_and_persist_pit_prices(universe_id=universe_id)
    return {"inserted": inserted}
