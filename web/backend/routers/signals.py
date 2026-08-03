from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.signal_publication_service import DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE
from web.backend.admin import require_admin
from web.backend.app_settings import PUBLISH_SIGNALS_ENABLED_KEY, get_setting_bool
from web.backend.db import service_conn
from web.backend.rate_limit import limiter
from web.backend.signal_publication import publish_daily_signals

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


@router.get("/published")
@limiter.limit("60/minute")
async def list_published_signals(
    request: Request,
    target_date: date | None = Query(None),
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
):
    """
    TR-5 groundwork: the public track record, unauthenticated by design —
    this is the whole point of "published". Defaults to the most recent
    publication for the given universe/lookback when no date is given.
    """
    async with service_conn() as conn:
        if target_date is None:
            target_date = await conn.fetchval(
                """
                SELECT max(target_date) FROM published_signals
                WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
                """,
                universe_id, lookback_days,
            )
            if target_date is None:
                return {"target_date": None, "universe_id": universe_id, "lookback_days": lookback_days, "signals": []}

        rows = await conn.fetch(
            """
            SELECT * FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank ASC
            """,
            target_date, universe_id, lookback_days,
        )

    return {
        "target_date": str(target_date),
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "signals": [_record_to_dict(r) for r in rows],
    }


@router.post("/publish-now", dependencies=[Depends(require_admin)])
async def publish_now(
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
    force: bool = Query(False, description="Bypass the publish_signals_enabled gate for a controlled test."),
):
    """Manual trigger for the same publication the scheduler runs daily —
    for verifying the pipeline and catching up a missed day, not routine use.
    Respects the same off-by-default gate as the scheduled job unless
    force=true is passed explicitly, so a routine test call can't
    accidentally become the record's real first publication."""
    if not force and not await get_setting_bool(PUBLISH_SIGNALS_ENABLED_KEY, default=False):
        raise HTTPException(
            409,
            "Publishing is currently disabled (publish_signals_enabled=false). "
            "Enable it via /admin/settings, or pass force=true for a one-off test.",
        )
    published = await publish_daily_signals(universe_id=universe_id, lookback_days=lookback_days)
    return {"published": published}
