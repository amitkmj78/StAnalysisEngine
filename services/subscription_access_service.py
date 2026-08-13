"""
Horizon 1 (RS-1, RS-2) — the single place that answers "is this user an
active paid subscriber." Deliberately tiny and deliberately the *only*
thing content-gating is allowed to check: RS-1's impersonality constraint
means gating may branch on subscription status and nothing else about
the subscriber (never portfolio, positions, risk profile, or any other
attribute) — keeping the check itself minimal makes that boundary harder
to accidentally violate later.
"""

from datetime import date, timedelta
from typing import Optional

from web.backend.db import service_conn


async def is_active_paid_subscriber(user_id: str) -> bool:
    async with service_conn() as conn:
        row = await conn.fetchval(
            """
            SELECT 1 FROM subscriptions
            WHERE user_id = $1::uuid AND tier = 'paid' AND status = 'active'
            LIMIT 1
            """,
            user_id,
        )
    return row is not None


def compute_free_tier_target_date(
    requested_date: Optional[date],
    latest_date: Optional[date],
    record_start_date: Optional[date],
    lag_days: int,
) -> tuple[Optional[date], bool]:
    """
    RS-2's "free tier exposes ... lagged rankings" — pure date math,
    factored out of the router so it's directly unit-testable. Returns
    (resolved_target_date, is_lagged).

    A too-recent or omitted requested_date is capped to
    latest_date - lag_days; a request for something already older than
    that cutoff passes through unchanged (a free reader can still browse
    old history freely, just not today's). Clamped to record_start_date
    so a track record younger than the lag window doesn't show a free
    reader nothing for its first few days.
    """
    if latest_date is None:
        return requested_date, False

    free_cutoff = latest_date - timedelta(days=lag_days)
    if record_start_date and free_cutoff < record_start_date:
        free_cutoff = record_start_date

    if requested_date is None or requested_date > free_cutoff:
        return free_cutoff, True
    return requested_date, True
