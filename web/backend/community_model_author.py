"""COM-7: the app's own model appears on the leaderboard as its own
author, scored by the same rules as everyone else (COM-7's own
acceptance text). Reuses services/signal_change_alert_service.py's
existing "did this ticker's short-term signal just change today"
detection -- same query shape, scoped to the WHOLE universe instead of
just owned/watchlisted tickers -- rather than re-deriving "is this a
fresh signal" a second time. The sentinel "model author" identity
(author_user_id NULL, is_model=true) mirrors services/
quant_model_service.py's MODEL_MEMBER_LABEL convention for the
challenges leaderboard -- no real users row, by design.
"""

import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.stock_detail_service import DET3_SHORT_HORIZON_DAYS
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

UNIVERSE_ID = "All"
# Not a real disclosed position -- the app itself never holds anything;
# this is what has_position/disclosure_note/attested_no_promotion mean
# for a model-authored idea, stated plainly rather than defaulted to
# something that would misleadingly resemble a human author's own
# attestation.
_MODEL_DISCLOSURE = "Model-generated from a fresh SCR-1 short-term signal change; the app holds no positions."


async def publish_model_ideas_for_today() -> int:
    """One idea per ticker whose short_signal freshly became Buy or
    Trim today (vs. yesterday's captured value) -- idempotent per
    (ticker, day) via an explicit existence check, since this can be
    safely re-run (scheduler restart, manual trigger)."""
    today = date.today()

    async with service_conn() as conn:
        latest_rows = await conn.fetch(
            """
            SELECT ticker, short_signal FROM stock_scores
            WHERE universe_id = $1 AND as_of_date = $2 AND short_signal IN ('Buy', 'Trim')
            """,
            UNIVERSE_ID, today,
        )
    if not latest_rows:
        return 0

    published = 0
    async with service_conn() as conn:
        for row in latest_rows:
            ticker, signal = row["ticker"], row["short_signal"]
            prior = await conn.fetchrow(
                """
                SELECT short_signal FROM stock_scores
                WHERE ticker = $1 AND universe_id = $2 AND as_of_date < $3
                ORDER BY as_of_date DESC LIMIT 1
                """,
                ticker, UNIVERSE_ID, today,
            )
            if prior is None or prior["short_signal"] == signal:
                continue  # not a fresh change, or no prior day to compare (same trap signal_change_alert_service.py already solved)

            already_published = await conn.fetchval(
                "SELECT 1 FROM community_ideas WHERE is_model AND ticker = $1 AND created_at::date = $2",
                ticker, today,
            )
            if already_published:
                continue

            entry_price = await run_in_threadpool(get_latest_price, ticker)
            if entry_price is None:
                continue

            direction = "LONG" if signal == "Buy" else "SHORT"
            await conn.execute(
                """
                INSERT INTO community_ideas (
                    author_user_id, is_model, ticker, direction, horizon_days, entry_price,
                    has_position, disclosure_note, attested_no_promotion
                ) VALUES (NULL, true, $1, $2, $3, $4, false, $5, true)
                """,
                ticker, direction, DET3_SHORT_HORIZON_DAYS, entry_price, _MODEL_DISCLOSURE,
            )
            published += 1

    if published:
        logger.info("Published %d model-authored community ideas", published)
    return published
