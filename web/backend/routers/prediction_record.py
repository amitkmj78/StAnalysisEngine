"""DIF-5: the public, chained record of the app's predictions.

GET is public: anyone can read the published days and recompute the hashes. Publishing is admin-only and
adds one day, the most recent day that has score rows. A day is never changed once published.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.prediction_hash_service import chain_hash, day_content_hash
from web.backend.admin import require_admin
from web.backend.db import service_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/record", tags=["prediction-record"])
UNIVERSE = "All"


@router.get("/hashes")
@limiter.limit("60/minute")
async def list_hashes(request: Request, limit: int = Query(90, ge=1, le=365)):
    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT day, record_count, content_hash, chain_hash, created_at FROM prediction_hash_chain ORDER BY day DESC LIMIT $1",
            limit,
        )
    return {
        "days": [
            {"day": r["day"].isoformat(), "record_count": r["record_count"], "content_hash": r["content_hash"],
             "chain_hash": r["chain_hash"], "published_at": r["created_at"].isoformat()}
            for r in rows
        ],
        "how_to_check": (
            "Each day's chain_hash = SHA-256 of the JSON {prev, day, count, content}, where prev is the previous "
            "day's chain_hash. content is the SHA-256 of that day's sorted prediction lines."
        ),
    }


@router.post("/publish", dependencies=[Depends(require_admin)])
async def publish_latest_day(request: Request):
    """Admin only. Adds the most recent day that has score rows to the chain, unless it is already there."""
    async with service_conn() as conn:
        latest = await conn.fetchval(
            "SELECT MAX(as_of_date) FROM stock_scores WHERE universe_id = $1", UNIVERSE
        )
        if latest is None:
            raise HTTPException(409, "There are no score rows to publish yet.")
        existing = await conn.fetchval("SELECT 1 FROM prediction_hash_chain WHERE day = $1", latest)
        if existing:
            return {"published": False, "day": latest.isoformat(), "reason": "already on the chain"}
        rows = await conn.fetch(
            """
            SELECT ticker, universe_id, as_of_date, short_score, short_signal, long_score, long_signal, sector_key,
                   (SELECT regime_confirmed FROM market_regime_daily WHERE as_of_date = s.as_of_date) AS regime
            FROM stock_scores s WHERE universe_id = $1 AND as_of_date = $2
            """,
            UNIVERSE, latest,
        )
        previous = await conn.fetchrow("SELECT chain_hash FROM prediction_hash_chain ORDER BY day DESC LIMIT 1")
        records = [dict(r) for r in rows]
        content = day_content_hash(records)
        chained = chain_hash(previous["chain_hash"] if previous else None, latest.isoformat(), len(records), content)
        await conn.execute(
            "INSERT INTO prediction_hash_chain (day, record_count, content_hash, chain_hash) VALUES ($1, $2, $3, $4)",
            latest, len(records), content, chained,
        )
    return {"published": True, "day": latest.isoformat(), "record_count": len(records), "chain_hash": chained}
