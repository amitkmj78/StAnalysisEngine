"""COM-1..7: published trade ideas, auto-scored at their horizon
against SPY, author profiles, a risk-adjusted leaderboard, follow +
alerts, and moderation (report button + a required position-disclosure
field). See services/community_idea_service.py and services/
community_leaderboard_service.py for the pure scoring/ranking logic
this router's queries feed.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.community_idea_service import DIRECTIONS, evaluate_idea_outcome
from services.community_leaderboard_service import build_leaderboard
from services.data_service import get_latest_price
from services.notification_dispatcher import dispatch_alert
from services.quant_model_service import MODEL_MEMBER_LABEL
from web.backend.admin import require_admin
from web.backend.auth import verify_bearer_token
from web.backend.community_ideas_eval import evaluate_due_community_ideas
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/community", tags=["community"])

# COM-6: an idea is auto-hidden (pending admin review, never deleted
# outright) once it accumulates this many DISTINCT reporters -- not a
# single report, so one person can't unilaterally hide another's idea.
AUTO_HIDE_REPORT_THRESHOLD = 3

MAX_HORIZON_DAYS = 252  # ~1 trading year, the app's own long-term horizon elsewhere (DET3_LONG_HORIZON_DAYS)
MIN_HORIZON_DAYS = 1


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


class IdeaCreateRequest(BaseModel):
    ticker: str
    direction: str
    horizon_days: int
    target: Optional[float] = None
    stop: Optional[float] = None
    has_position: bool
    disclosure_note: Optional[str] = None
    attested_no_promotion: bool


@router.post("/ideas", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("20/minute")
async def create_idea(request: Request, body: IdeaCreateRequest):
    """COM-1: publish a LOCKED idea -- there is no edit endpoint, ever;
    only COM-6's moderation can hide one later. COM-6: has_position and
    attested_no_promotion are REQUIRED, never optional -- the real,
    honest moderation mechanism this feature has (no automated pump-
    and-dump/paid-promotion detection exists or realistically can)."""
    await enforce_daily_quota(request, "community/ideas/create")
    user_id = request.state.user["id"]

    if body.direction not in DIRECTIONS:
        raise HTTPException(422, f"direction must be one of {sorted(DIRECTIONS)}")
    if not MIN_HORIZON_DAYS <= body.horizon_days <= MAX_HORIZON_DAYS:
        raise HTTPException(422, f"horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}")
    if not body.attested_no_promotion:
        raise HTTPException(422, "You must attest this idea is not paid promotion before publishing.")
    ticker = body.ticker.strip().upper()
    if not ticker:
        raise HTTPException(422, "ticker is required")

    async with service_conn() as conn:
        display_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", user_id)
    if not display_name:
        raise HTTPException(422, "Set a display name (PUT /api/v1/auth/me/display-name) before publishing an idea.")

    entry_price = await run_in_threadpool(get_latest_price, ticker)
    if entry_price is None:
        raise HTTPException(422, f"Could not get a live price for {ticker}.")

    async with service_conn() as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO community_ideas (
                author_user_id, ticker, direction, horizon_days, target, stop, entry_price,
                has_position, disclosure_note, attested_no_promotion
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            user_id, ticker, body.direction, body.horizon_days, body.target, body.stop, entry_price,
            body.has_position, body.disclosure_note, body.attested_no_promotion,
        )
        # COM-5: fan out to every follower of this author -- an in-app
        # alert (followed_author_alerts, the 8th alerts_inbox.py UNION
        # branch) plus whatever channels dispatch_alert resolves
        # (email/push/webhook per that follower's own preferences).
        followers = await conn.fetch(
            "SELECT follower_user_id FROM author_follows WHERE followed_user_id = $1::uuid", user_id
        )
        for f in followers:
            await conn.execute(
                """
                INSERT INTO followed_author_alerts (user_id, author_user_id, idea_id, ticker)
                VALUES ($1::uuid, $2::uuid, $3, $4)
                """,
                f["follower_user_id"], user_id, record["id"], ticker,
            )
    for f in followers:
        await dispatch_alert(
            str(f["follower_user_id"]), ticker, "followed_author_idea",
            f"{display_name} published a new idea: {ticker}",
            f"{display_name} published a {body.direction} idea on {ticker} (horizon: {body.horizon_days} days).",
            {"author_display_name": display_name, "direction": body.direction},
        )
    return _record_to_dict(record)


@router.get("/ideas")
async def list_ideas(limit: int = Query(50, le=200)):
    """COM-1's public feed -- excludes hidden (COM-6) ideas."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.*, u.display_name
            FROM community_ideas ci
            LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE NOT ci.hidden
            ORDER BY ci.created_at DESC
            LIMIT $1
            """,
            limit,
        )
    return {"ideas": [{**_record_to_dict(r), "display_name": r["display_name"] or MODEL_MEMBER_LABEL} for r in rows]}


@router.post("/ideas/evaluate-now", dependencies=[Depends(require_admin)])
async def evaluate_ideas_now():
    """COM-2: manual trigger for the same scoring the scheduler runs
    daily -- for verifying the pipeline, not routine use. No enable-
    gate, same reasoning /signals/evaluate-now already uses: scoring
    an already-published idea isn't itself a new disclosure."""
    scored = await evaluate_due_community_ideas()
    return {"scored": scored}


@router.get("/authors/{author_id}")
async def get_author_profile(author_id: str):
    """COM-3: a public author profile -- display name, ideas count,
    hit rate, average excess return, worst idea, and the full idea
    list. `author_id == "model"` is COM-7's sentinel path for the
    app's own model author (no real users row exists for it)."""
    async with service_conn() as conn:
        if author_id == "model":
            rows = await conn.fetch(
                "SELECT * FROM community_ideas WHERE is_model AND NOT hidden ORDER BY created_at DESC"
            )
            display_name = MODEL_MEMBER_LABEL
        else:
            rows = await conn.fetch(
                "SELECT * FROM community_ideas WHERE author_user_id = $1::uuid AND NOT hidden ORDER BY created_at DESC",
                author_id,
            )
            display_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", author_id)
            if display_name is None and not rows:
                raise HTTPException(404, "Author not found.")

    ideas = [_record_to_dict(r) for r in rows]
    scored = [i for i in ideas if i["scored_at"] is not None]
    board = build_leaderboard(
        [{**i, "is_model": author_id == "model", "author_user_id": author_id, "display_name": display_name} for i in scored]
    ) if scored else []
    summary = board[0] if board else {
        "num_ideas": len(scored), "avg_excess_vs_spy_pct": None, "hit_rate_pct": None, "score": None, "worst_idea": None,
    }
    return {
        "author_id": author_id,
        "display_name": display_name or MODEL_MEMBER_LABEL,
        "num_ideas_total": len(ideas),
        "num_ideas_scored": len(scored),
        **{k: summary[k] for k in ("avg_excess_vs_spy_pct", "hit_rate_pct", "score", "worst_idea")},
        "ideas": ideas,
    }


@router.get("/leaderboard")
async def get_leaderboard():
    """COM-4: ranked by risk-adjusted excess return with a minimum
    sample (services.community_idea_service.MIN_IDEAS_FOR_LEADERBOARD)
    -- never raw return alone, and an author below the sample gate is
    still listed (return + sample size shown), just not ranked by a
    score the data can't support."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.author_user_id, ci.is_model, ci.direction, ci.realized_return_pct,
                   ci.excess_vs_spy_pct, ci.outcome, u.display_name
            FROM community_ideas ci
            LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE ci.scored_at IS NOT NULL AND NOT ci.hidden
            """
        )
    return {"leaderboard": build_leaderboard([_record_to_dict(r) for r in rows])}


@router.post("/authors/{author_id}/follow", dependencies=[Depends(verify_bearer_token)])
async def follow_author(request: Request, author_id: str):
    follower_id = request.state.user["id"]
    if follower_id == author_id:
        raise HTTPException(422, "You can't follow yourself.")
    async with service_conn() as conn:
        inserted = await conn.fetchrow(
            """
            INSERT INTO author_follows (follower_user_id, followed_user_id)
            VALUES ($1::uuid, $2::uuid)
            ON CONFLICT (follower_user_id, followed_user_id) DO NOTHING
            RETURNING id
            """,
            follower_id, author_id,
        )
        # SOC-9: a new_follower notification -- only on a genuinely new
        # follow (ON CONFLICT DO NOTHING above returns no row on a
        # repeat follow), so re-following never double-notifies.
        if inserted is not None:
            follower_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", follower_id)
            summary = f"{follower_name or 'Someone'} started following you"
            await conn.execute(
                """
                INSERT INTO social_notifications (user_id, notification_type, actor_user_id, summary)
                VALUES ($1::uuid, 'new_follower', $2::uuid, $3)
                """,
                author_id, follower_id, summary,
            )
    if inserted is not None:
        await dispatch_alert(author_id, None, "new_follower", summary, f"{follower_name or 'Someone'} started following you.")
    return {"ok": True}


@router.delete("/authors/{author_id}/follow", dependencies=[Depends(verify_bearer_token)])
async def unfollow_author(request: Request, author_id: str):
    follower_id = request.state.user["id"]
    async with service_conn() as conn:
        await conn.execute(
            "DELETE FROM author_follows WHERE follower_user_id = $1::uuid AND followed_user_id = $2::uuid",
            follower_id, author_id,
        )
    return {"ok": True}


class ReportRequest(BaseModel):
    reason: str


@router.post("/ideas/{idea_id}/report", dependencies=[Depends(verify_bearer_token)])
@limiter.limit("20/minute")
async def report_idea(request: Request, idea_id: int, body: ReportRequest):
    """COM-6: the report button. One report per (idea, reporter) --
    a repeat click never inflates the auto-hide count. Auto-hides
    (pending admin review, never deleted) once distinct reporters
    cross AUTO_HIDE_REPORT_THRESHOLD."""
    reporter_id = request.state.user["id"]
    reason = body.reason.strip()
    if not reason:
        raise HTTPException(422, "reason is required")
    async with service_conn() as conn:
        exists = await conn.fetchval("SELECT 1 FROM community_ideas WHERE id = $1", idea_id)
        if not exists:
            raise HTTPException(404, "Idea not found.")
        await conn.execute(
            """
            INSERT INTO idea_reports (idea_id, reporter_user_id, reason)
            VALUES ($1, $2::uuid, $3)
            ON CONFLICT (idea_id, reporter_user_id) DO NOTHING
            """,
            idea_id, reporter_id, reason,
        )
        report_count = await conn.fetchval("SELECT count(*) FROM idea_reports WHERE idea_id = $1", idea_id)
        if report_count >= AUTO_HIDE_REPORT_THRESHOLD:
            await conn.execute("UPDATE community_ideas SET hidden = true WHERE id = $1", idea_id)
    return {"ok": True, "report_count": report_count, "hidden": report_count >= AUTO_HIDE_REPORT_THRESHOLD}


@router.get("/admin/reports", dependencies=[Depends(require_admin)])
async def list_reported_ideas():
    """COM-6's admin review queue -- every idea with at least one
    report, most-reported first, for an admin to restore or remove."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ci.*, u.display_name, count(r.id) AS report_count,
                   array_agg(r.reason) AS reasons
            FROM community_ideas ci
            JOIN idea_reports r ON r.idea_id = ci.id
            LEFT JOIN users u ON u.id = ci.author_user_id
            GROUP BY ci.id, u.display_name
            ORDER BY report_count DESC, ci.created_at DESC
            """
        )
    return {"ideas": [_record_to_dict(r) for r in rows]}


@router.post("/admin/ideas/{idea_id}/restore", dependencies=[Depends(require_admin)])
async def restore_idea(idea_id: int):
    async with service_conn() as conn:
        row = await conn.fetchrow("UPDATE community_ideas SET hidden = false WHERE id = $1 RETURNING id", idea_id)
    if row is None:
        raise HTTPException(404, "Idea not found.")
    return {"ok": True}


@router.delete("/admin/ideas/{idea_id}", dependencies=[Depends(require_admin)])
async def delete_idea(idea_id: int):
    async with service_conn() as conn:
        row = await conn.fetchrow("DELETE FROM community_ideas WHERE id = $1 RETURNING id", idea_id)
    if row is None:
        raise HTTPException(404, "Idea not found.")
    return {"ok": True}
