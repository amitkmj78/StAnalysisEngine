"""SOC-1: nightly recompute of users.verified_badge -- NEVER user-
settable (see migrations/2026-10-social_network.sql), backed by real
COM-3/COM-4 scored-idea track record: at least
community_idea_service.MIN_IDEAS_FOR_LEADERBOARD scored ideas with a
positive risk-adjusted score. Reuses COM-4's own
community_leaderboard_service.build_leaderboard, not a second scoring
path. One UPDATE, idempotent, only touches rows whose badge actually
changes (earned or lost) -- a badge is revoked if a track record no
longer supports it, same as it's only granted once it does.
"""

from __future__ import annotations

from services.community_idea_service import MIN_IDEAS_FOR_LEADERBOARD
from services.community_leaderboard_service import build_leaderboard
from web.backend.db import service_conn

# BEG-5: a disclosed, chosen threshold (not derived from data) -- same
# style as MIN_IDEAS_FOR_LEADERBOARD above.
MENTOR_MIN_ACCEPTED_ANSWERS = 3


async def recompute_verified_badges() -> int:
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.author_user_id, ci.is_model, ci.direction, ci.realized_return_pct,
                   ci.excess_vs_spy_pct, ci.outcome, u.display_name
            FROM community_ideas ci
            LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE ci.scored_at IS NOT NULL AND NOT ci.hidden AND NOT ci.is_model
            """
        )
        board = build_leaderboard([dict(r) for r in rows], min_samples=MIN_IDEAS_FOR_LEADERBOARD)
        earners = [e["author_user_id"] for e in board if e["score"] is not None and e["score"] > 0]

        changed = await conn.fetch(
            """
            UPDATE users SET verified_badge = (id = ANY($1::uuid[]))
            WHERE verified_badge IS DISTINCT FROM (id = ANY($1::uuid[]))
            RETURNING id
            """,
            earners,
        )
    return len(changed)


async def recompute_mentor_badges() -> int:
    """BEG-5: mentor_badge = verified_badge (COM-3) AND at least
    MENTOR_MIN_ACCEPTED_ANSWERS accepted answers (BEG-4). Same idempotent,
    only-touch-what-changed shape as recompute_verified_badges above --
    revoked as readily as it's granted if either condition stops holding."""
    async with service_conn() as conn:
        earners = await conn.fetch(
            """
            SELECT u.id FROM users u
            WHERE u.verified_badge AND (
                SELECT count(*) FROM post_comments c WHERE c.author_user_id = u.id AND c.is_accepted
            ) >= $1
            """,
            MENTOR_MIN_ACCEPTED_ANSWERS,
        )
        earner_ids = [r["id"] for r in earners]
        changed = await conn.fetch(
            """
            UPDATE users SET mentor_badge = (id = ANY($1::uuid[]))
            WHERE mentor_badge IS DISTINCT FROM (id = ANY($1::uuid[]))
            RETURNING id
            """,
            earner_ids,
        )
    return len(changed)
