"""SOC-1..9: profiles (experience level/interests/a system-verified
badge), a followed-people/tickers/topics feed of posts + chart
snapshots + community ideas, per-ticker discussion, shareable charts,
claim verification, groups with moderators, polling chat rooms +
permissioned DMs, and a reputation score. See services/
social_feed_service.py, services/reputation_service.py and services/
social_chat_service.py for the pure logic this router's queries feed.

Scope cuts, disclosed in the tracker (docs/
StAnalysisEngine_Requirements_Tracker.html): SOC-4's shared chart is a
fork-to-edit copy, not live co-editing; SOC-5's claim verification is
an explicit author-chosen link to one of their own community_ideas,
not NLP claim detection; SOC-6's private-group visibility is enforced
here in the router, not by a Postgres RLS policy; SOC-7's chat is
polling-based (no WebSocket layer exists anywhere in this app) and has
no market-holiday calendar; SOC-8's reputation omits the "accepted
answers" component, blocked on the separate, not-yet-built BEG-4 (Q&A
forum).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.notification_dispatcher import dispatch_alert
from services.reputation_service import compute_reputation
from services.social_chat_service import is_market_hours_now, validate_room
from services.social_feed_service import build_feed
from web.backend.admin import require_admin
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.rate_limit import limiter

router = APIRouter(prefix="/api/v1/social", tags=["social"], dependencies=[Depends(verify_bearer_token)])

# BEG-4/5: how many of a user's answers have been accepted by the asker --
# feeds reputation's qa_component and the mentor_badge nightly recompute.
_ACCEPTED_ANSWER_COUNT_SQL = "SELECT count(*) FROM post_comments WHERE author_user_id = $1::uuid AND is_accepted"

EXPERIENCE_LEVELS = {"beginner", "intermediate", "experienced"}
MAX_INTERESTS = 10
MAX_POST_BODY = 2000
MAX_COMMENT_BODY = 1000
MAX_CHAT_BODY = 500
_MENTION_PATTERN = re.compile(r"@(\w{2,30})")
_TICKER_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


def _ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not _TICKER_PATTERN.match(t):
        raise HTTPException(422, "Not a valid ticker.")
    return t


async def _dispatch_mentions(conn, body: str, actor_id: str, actor_name: str, post_id: Optional[int]) -> None:
    """A simple @handle regex against users.display_name -- not NLP,
    and only matches a display name that is itself a single \\w token
    (disclosed limitation)."""
    handles = {m.group(1) for m in _MENTION_PATTERN.finditer(body)}
    if not handles:
        return
    rows = await conn.fetch("SELECT id, display_name FROM users WHERE display_name = ANY($1::text[])", list(handles))
    for r in rows:
        if str(r["id"]) == actor_id:
            continue
        summary = f"{actor_name} mentioned you"
        await conn.execute(
            """
            INSERT INTO social_notifications (user_id, notification_type, actor_user_id, post_id, summary)
            VALUES ($1::uuid, 'mention', $2::uuid, $3, $4)
            """,
            r["id"], actor_id, post_id, summary,
        )
        await dispatch_alert(str(r["id"]), None, "mention", summary, f"{actor_name} mentioned you in a post.")


# ---------------------------------------------------------------- SOC-1: profiles

class ProfileUpdateRequest(BaseModel):
    experience_level: Optional[str] = None
    interests: Optional[list[str]] = None


@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "SELECT id, display_name, experience_level, interests, verified_badge, mentor_badge FROM users WHERE id = $1::uuid",
            user_id,
        )
        if row is None:
            raise HTTPException(404, "User not found.")
        idea_rows = await conn.fetch(
            """
            SELECT author_user_id, is_model, direction, realized_return_pct, excess_vs_spy_pct, outcome, display_name
            FROM community_ideas ci LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE author_user_id = $1::uuid AND scored_at IS NOT NULL AND NOT hidden
            """,
            user_id,
        )
        accepted_answer_count = await conn.fetchval(_ACCEPTED_ANSWER_COUNT_SQL, user_id)
    reputation = compute_reputation([dict(r) for r in idea_rows], accepted_answer_count)
    return {**_record_to_dict(row), "reputation": reputation}


@router.put("/profile")
async def update_profile(request: Request, body: ProfileUpdateRequest):
    """Self-only. verified_badge is never settable here -- see
    web/backend/social_badges.py's nightly recompute."""
    user_id = request.state.user["id"]
    if body.experience_level is not None and body.experience_level not in EXPERIENCE_LEVELS:
        raise HTTPException(422, f"experience_level must be one of {sorted(EXPERIENCE_LEVELS)}")
    interests = body.interests
    if interests is not None:
        interests = [i.strip() for i in interests if i.strip()][:MAX_INTERESTS]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE users SET
                experience_level = COALESCE($2, experience_level),
                interests = COALESCE($3::jsonb, interests)
            WHERE id = $1::uuid
            RETURNING id, display_name, experience_level, interests, verified_badge, mentor_badge
            """,
            user_id, body.experience_level, None if interests is None else json.dumps(interests),
        )
    return _record_to_dict(row)


@router.get("/reputation/{user_id}")
async def get_reputation(user_id: str):
    async with service_conn() as conn:
        idea_rows = await conn.fetch(
            """
            SELECT author_user_id, is_model, direction, realized_return_pct, excess_vs_spy_pct, outcome, display_name
            FROM community_ideas ci LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE author_user_id = $1::uuid AND scored_at IS NOT NULL AND NOT hidden
            """,
            user_id,
        )
        accepted_answer_count = await conn.fetchval(_ACCEPTED_ANSWER_COUNT_SQL, user_id)
    return compute_reputation([dict(r) for r in idea_rows], accepted_answer_count)


# ---------------------------------------------------------------- SOC-2: ticker/topic follows

@router.post("/tickers/{ticker}/follow")
async def follow_ticker(request: Request, ticker: str):
    user_id = request.state.user["id"]
    ticker = _ticker(ticker)
    async with service_conn() as conn:
        await conn.execute(
            "INSERT INTO ticker_follows (user_id, ticker) VALUES ($1::uuid, $2) ON CONFLICT DO NOTHING",
            user_id, ticker,
        )
    return {"ok": True}


@router.delete("/tickers/{ticker}/follow")
async def unfollow_ticker(request: Request, ticker: str):
    user_id = request.state.user["id"]
    ticker = _ticker(ticker)
    async with service_conn() as conn:
        await conn.execute("DELETE FROM ticker_follows WHERE user_id = $1::uuid AND ticker = $2", user_id, ticker)
    return {"ok": True}


@router.post("/topics/{topic}/follow")
async def follow_topic(request: Request, topic: str):
    user_id = request.state.user["id"]
    topic = topic.strip().lower()
    if not topic:
        raise HTTPException(422, "topic is required")
    async with service_conn() as conn:
        await conn.execute(
            "INSERT INTO topic_follows (user_id, topic) VALUES ($1::uuid, $2) ON CONFLICT DO NOTHING",
            user_id, topic,
        )
    return {"ok": True}


@router.delete("/topics/{topic}/follow")
async def unfollow_topic(request: Request, topic: str):
    user_id = request.state.user["id"]
    topic = topic.strip().lower()
    async with service_conn() as conn:
        await conn.execute("DELETE FROM topic_follows WHERE user_id = $1::uuid AND topic = $2", user_id, topic)
    return {"ok": True}


# ---------------------------------------------------------------- SOC-2: feed

@router.get("/feed")
async def get_feed(request: Request, limit: int = Query(50, le=200)):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        followed_people = await conn.fetch("SELECT followed_user_id FROM author_follows WHERE follower_user_id = $1::uuid", user_id)
        followed_tickers = await conn.fetch("SELECT ticker FROM ticker_follows WHERE user_id = $1::uuid", user_id)
        followed_topics = await conn.fetch("SELECT topic FROM topic_follows WHERE user_id = $1::uuid", user_id)
        people_ids = [r["followed_user_id"] for r in followed_people] + [user_id]
        tickers = [r["ticker"] for r in followed_tickers]
        topics = [r["topic"] for r in followed_topics]

        post_rows = await conn.fetch(
            """
            SELECT p.*, u.display_name FROM posts p LEFT JOIN users u ON u.id = p.author_user_id
            WHERE NOT p.hidden AND (
                p.author_user_id = ANY($1::uuid[]) OR p.ticker = ANY($2::text[]) OR p.topic = ANY($3::text[])
            )
            ORDER BY p.created_at DESC LIMIT $4
            """,
            people_ids, tickers, topics, limit,
        )
        idea_rows = await conn.fetch(
            """
            SELECT ci.*, u.display_name FROM community_ideas ci LEFT JOIN users u ON u.id = ci.author_user_id
            WHERE NOT ci.hidden AND (
                ci.author_user_id = ANY($1::uuid[]) OR ci.ticker = ANY($2::text[])
            )
            ORDER BY ci.created_at DESC LIMIT $3
            """,
            people_ids, tickers, limit,
        )
    feed = build_feed([_record_to_dict(r) for r in post_rows], [_record_to_dict(r) for r in idea_rows], limit=limit)
    return {"feed": feed}


# ---------------------------------------------------------------- SOC-2/3/4/5: posts

class PostCreateRequest(BaseModel):
    body: str
    post_type: str = "note"
    ticker: Optional[str] = None
    topic: Optional[str] = None
    group_id: Optional[int] = None
    # STS-6 (comments/questions half only -- ratings deferred, see
    # strategy_builder.py's module docstring): a published strategy's
    # discussion, same openness as a ticker-page post -- no membership
    # gate, unlike group_id above.
    published_strategy_id: Optional[int] = None
    attach_chart: bool = False
    claim_reference_id: Optional[int] = None
    # SAF-4: required whenever `ticker` is set -- self-declared unless the
    # author has a real Plaid-synced position in that exact ticker, in
    # which case the backend overrides it to a verified "true" regardless
    # of what was submitted (never trusts a client-claimed "verified").
    has_position: Optional[bool] = None
    # SAF-5: self-declared, honor system -- same shape as
    # community_ideas.attested_no_promotion.
    is_paid_promotion: bool = False


@router.post("/posts")
@limiter.limit("20/minute")
async def create_post(request: Request, body: PostCreateRequest):
    user_id = request.state.user["id"]
    text = body.body.strip()
    if not text or len(text) > MAX_POST_BODY:
        raise HTTPException(422, f"body must be 1-{MAX_POST_BODY} characters")
    if body.post_type not in ("note", "performance_claim", "question"):
        raise HTTPException(422, "post_type must be 'note', 'performance_claim' or 'question'")
    ticker = _ticker(body.ticker) if body.ticker else None
    topic = body.topic.strip().lower() if body.topic else None

    # SAF-4: required (not skippable) whenever a post sets a ticker --
    # mirrors community_ideas.has_position's existing required-field
    # convention.
    has_position = body.has_position
    if ticker and has_position is None:
        raise HTTPException(422, "has_position is required when posting about a ticker.")

    async with service_conn() as conn:
        display_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", user_id)
        if not display_name:
            raise HTTPException(422, "Set a display name (PUT /api/v1/auth/me/display-name) before posting.")

        # SAF-4: a real Plaid-synced position in this exact ticker overrides
        # the self-declaration to a verified "true" -- never the reverse,
        # and never trusts a client-claimed "verified" flag (there isn't
        # one; this is computed server-side from the author's own synced
        # holdings).
        position_verified = False
        if ticker:
            has_real_position = await conn.fetchval(
                """
                SELECT 1 FROM portfolio_positions
                WHERE user_id = $1::uuid AND ticker = $2 AND plaid_item_id IS NOT NULL LIMIT 1
                """,
                user_id, ticker,
            )
            if has_real_position:
                has_position = True
                position_verified = True

        if body.group_id is not None:
            is_member = await conn.fetchval(
                "SELECT 1 FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", body.group_id, user_id
            )
            if not is_member:
                raise HTTPException(403, "You must join this group before posting in it.")

        # SOC-5: a performance_claim is `verified` only when it links one
        # of the AUTHOR'S OWN community_ideas -- never auto-verified, and
        # never another author's idea passed off as this author's claim.
        verified = False
        claim_reference_id = None
        if body.post_type == "performance_claim" and body.claim_reference_id is not None:
            owns_idea = await conn.fetchval(
                "SELECT 1 FROM community_ideas WHERE id = $1 AND author_user_id = $2::uuid",
                body.claim_reference_id, user_id,
            )
            if not owns_idea:
                raise HTTPException(422, "claim_reference_id must be one of your own published ideas.")
            claim_reference_id = body.claim_reference_id
            verified = True

        # SOC-4: fork a frozen copy of the author's current chart_drawings
        # for this ticker -- not live co-editing (disclosed in the tracker).
        chart_snapshot_id = None
        if body.attach_chart:
            if not ticker:
                raise HTTPException(422, "attach_chart requires a ticker.")
            drawing_rows = await conn.fetch(
                "SELECT kind, points, text FROM chart_drawings WHERE user_id = $1::uuid AND ticker = $2",
                user_id, ticker,
            )
            if not drawing_rows:
                raise HTTPException(422, f"You have no saved chart drawings for {ticker} to attach.")

            payload = [{"kind": r["kind"], "points": json.loads(r["points"]) if isinstance(r["points"], str) else r["points"], "text": r["text"]} for r in drawing_rows]
            snapshot = await conn.fetchrow(
                "INSERT INTO chart_snapshots (ticker, created_by_user_id, drawings) VALUES ($1, $2::uuid, $3::jsonb) RETURNING id",
                ticker, user_id, json.dumps(payload),
            )
            chart_snapshot_id = snapshot["id"]

        record = await conn.fetchrow(
            """
            INSERT INTO posts (author_user_id, post_type, body, ticker, topic, group_id, chart_snapshot_id, claim_reference_id, verified, published_strategy_id, has_position, position_verified, is_paid_promotion)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING *
            """,
            user_id, body.post_type, text, ticker, topic, body.group_id, chart_snapshot_id, claim_reference_id, verified,
            body.published_strategy_id, has_position, position_verified, body.is_paid_promotion,
        )
        await _dispatch_mentions(conn, text, user_id, display_name, record["id"])

        if body.group_id is not None:
            members = await conn.fetch(
                "SELECT user_id FROM group_members WHERE group_id = $1 AND user_id <> $2::uuid", body.group_id, user_id
            )
            group_name = await conn.fetchval("SELECT name FROM groups WHERE id = $1", body.group_id)
            for m in members:
                summary = f"{display_name} posted in {group_name}"
                await conn.execute(
                    """
                    INSERT INTO social_notifications (user_id, notification_type, actor_user_id, post_id, group_id, summary)
                    VALUES ($1::uuid, 'group_activity', $2::uuid, $3, $4, $5)
                    """,
                    m["user_id"], user_id, record["id"], body.group_id, summary,
                )
    return {**_record_to_dict(record), "display_name": display_name}


@router.get("/posts")
async def list_posts(
    request: Request,
    ticker: Optional[str] = None, topic: Optional[str] = None,
    group_id: Optional[int] = None, author_id: Optional[str] = None,
    published_strategy_id: Optional[int] = None,
    limit: int = Query(50, le=200),
):
    """SOC-3's ticker-page discussion panel is this same endpoint,
    filtered by `ticker` -- no separate "discussion" table/endpoint. STS-6
    (comments/questions half only) reuses it the same way, filtered by
    `published_strategy_id`. SAF-7: always excludes posts from a user the
    VIEWER has blocked, and posts matching the viewer's own muted
    users/tickers/keywords -- scoped to the requesting viewer, never a
    global filter."""
    viewer_id = request.state.user["id"]
    where = ["NOT p.hidden"]
    params: list = [viewer_id]
    if ticker:
        params.append(_ticker(ticker))
        where.append(f"p.ticker = ${len(params)}")
    if topic:
        params.append(topic.strip().lower())
        where.append(f"p.topic = ${len(params)}")
    if group_id is not None:
        params.append(group_id)
        where.append(f"p.group_id = ${len(params)}")
    if published_strategy_id is not None:
        params.append(published_strategy_id)
        where.append(f"p.published_strategy_id = ${len(params)}")
    if author_id:
        params.append(author_id)
        where.append(f"p.author_user_id = ${len(params)}::uuid")
    where.append("NOT EXISTS (SELECT 1 FROM user_blocks ub WHERE ub.blocker_user_id = $1::uuid AND ub.blocked_user_id = p.author_user_id)")
    where.append(
        """NOT EXISTS (
            SELECT 1 FROM user_mutes um WHERE um.user_id = $1::uuid AND (
                (um.mute_type = 'user' AND um.mute_value = p.author_user_id::text)
                OR (um.mute_type = 'ticker' AND um.mute_value = p.ticker)
                OR (um.mute_type = 'keyword' AND p.body ILIKE '%' || um.mute_value || '%')
            )
        )"""
    )
    params.append(limit)
    async with service_conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT p.*, u.display_name FROM posts p LEFT JOIN users u ON u.id = p.author_user_id
            WHERE {' AND '.join(where)} ORDER BY p.created_at DESC LIMIT ${len(params)}
            """,
            *params,
        )
    return {"posts": [_record_to_dict(r) for r in rows]}


# SAF-2: one report per (content_type, content_id, reporter) -- a repeat
# click never inflates the auto-hide count. Covers posts/comments/groups/
# profiles; idea_reports (community.py) already covers community_ideas.
AUTO_HIDE_CONTENT_REPORT_THRESHOLD = 3
_REPORTABLE_TABLES = {"post": "posts", "comment": "post_comments", "group": "groups"}


class ContentReportRequest(BaseModel):
    content_type: str
    # A post/comment/group id as a string ("42"), or a profile's user id
    # (a uuid string) -- one field covers both, see content_reports.content_id.
    content_id: str
    reason: str


@router.post("/report")
@limiter.limit("20/minute")
async def report_content(request: Request, body: ContentReportRequest):
    if body.content_type not in ("post", "comment", "group", "profile"):
        raise HTTPException(422, "content_type must be 'post', 'comment', 'group' or 'profile'.")
    reason = body.reason.strip()
    if not reason:
        raise HTTPException(422, "reason is required.")
    reporter_id = request.state.user["id"]

    # Validated before ever opening a connection -- a malformed content_id
    # is a 422, not a wasted DB round trip.
    numeric_id: Optional[int] = None
    if body.content_type != "profile":
        try:
            numeric_id = int(body.content_id)
        except ValueError:
            raise HTTPException(422, "content_id must be a numeric id for this content_type.")

    async with service_conn() as conn:
        if body.content_type == "profile":
            try:
                exists = await conn.fetchval("SELECT 1 FROM users WHERE id = $1::uuid", body.content_id)
            except Exception:  # noqa: BLE001 -- a malformed uuid is a 422, not a 500
                raise HTTPException(422, "content_id must be a valid user id for content_type 'profile'.")
        else:
            table = _REPORTABLE_TABLES[body.content_type]
            exists = await conn.fetchval(f"SELECT 1 FROM {table} WHERE id = $1", numeric_id)
        if not exists:
            raise HTTPException(404, f"{body.content_type.capitalize()} not found.")
        await conn.execute(
            """
            INSERT INTO content_reports (content_type, content_id, reporter_user_id, reason)
            VALUES ($1, $2, $3::uuid, $4)
            ON CONFLICT (content_type, content_id, reporter_user_id) DO NOTHING
            """,
            body.content_type, body.content_id, reporter_id, reason,
        )
        report_count = await conn.fetchval(
            "SELECT count(*) FROM content_reports WHERE content_type = $1 AND content_id = $2",
            body.content_type, body.content_id,
        )
        hidden = report_count >= AUTO_HIDE_CONTENT_REPORT_THRESHOLD
        if hidden and body.content_type in ("post", "group"):
            table = _REPORTABLE_TABLES[body.content_type]
            await conn.execute(f"UPDATE {table} SET hidden = true WHERE id = $1", int(body.content_id))
    return {"ok": True, "report_count": report_count, "hidden": hidden}


# SAF-7: block (mutual DM/visibility) and mute (ticker/keyword/user, feed
# filtering only -- muting never blocks DMs or notifies the muted party).

class BlockRequest(BaseModel):
    user_id: str


@router.post("/blocks")
async def block_user(request: Request, body: BlockRequest):
    blocker_id = request.state.user["id"]
    if blocker_id == body.user_id:
        raise HTTPException(422, "You can't block yourself.")
    async with user_conn(blocker_id) as conn:
        await conn.execute(
            "INSERT INTO user_blocks (blocker_user_id, blocked_user_id) VALUES ($1::uuid, $2::uuid) ON CONFLICT DO NOTHING",
            blocker_id, body.user_id,
        )
    return {"ok": True}


@router.delete("/blocks/{blocked_user_id}")
async def unblock_user(request: Request, blocked_user_id: str):
    blocker_id = request.state.user["id"]
    async with user_conn(blocker_id) as conn:
        await conn.execute(
            "DELETE FROM user_blocks WHERE blocker_user_id = $1::uuid AND blocked_user_id = $2::uuid",
            blocker_id, blocked_user_id,
        )
    return {"ok": True}


@router.get("/blocks")
async def list_blocks(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            """
            SELECT b.blocked_user_id, u.display_name FROM user_blocks b
            LEFT JOIN users u ON u.id = b.blocked_user_id
            WHERE b.blocker_user_id = $1::uuid ORDER BY b.created_at DESC
            """,
            user_id,
        )
    return {"blocks": [{"user_id": str(r["blocked_user_id"]), "display_name": r["display_name"]} for r in rows]}


class MuteRequest(BaseModel):
    mute_type: str
    mute_value: str


@router.post("/mutes")
async def mute(request: Request, body: MuteRequest):
    if body.mute_type not in ("user", "ticker", "keyword"):
        raise HTTPException(422, "mute_type must be 'user', 'ticker' or 'keyword'.")
    value = body.mute_value.strip()
    if body.mute_type == "ticker":
        value = _ticker(value)
    elif not value:
        raise HTTPException(422, "mute_value is required.")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        await conn.execute(
            "INSERT INTO user_mutes (user_id, mute_type, mute_value) VALUES ($1::uuid, $2, $3) ON CONFLICT DO NOTHING",
            user_id, body.mute_type, value,
        )
    return {"ok": True}


@router.delete("/mutes/{mute_id}")
async def unmute(request: Request, mute_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM user_mutes WHERE id = $1 AND user_id = $2::uuid RETURNING id", mute_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Mute not found.")
    return {"ok": True}


@router.get("/mutes")
async def list_mutes(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT id, mute_type, mute_value, created_at FROM user_mutes WHERE user_id = $1::uuid ORDER BY created_at DESC",
            user_id,
        )
    return {"mutes": [_record_to_dict(r) for r in rows]}


# SAF-2: admin review queue for content_reports (posts/comments/groups/
# profiles) -- idea_reports keeps its own separate admin queue in
# community.py, since community_ideas' row shape doesn't union cleanly
# with these four very different content types.

@router.get("/admin/reports", dependencies=[Depends(require_admin)])
async def list_content_reports(resolved: bool = False):
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT cr.*, u.display_name AS reporter_display_name
            FROM content_reports cr LEFT JOIN users u ON u.id = cr.reporter_user_id
            WHERE ($1 AND cr.reviewed_at IS NOT NULL) OR (NOT $1 AND cr.reviewed_at IS NULL)
            ORDER BY cr.created_at DESC
            """,
            resolved,
        )
    return {"reports": [_record_to_dict(r) for r in rows]}


class ResolveReportRequest(BaseModel):
    resolution: str


@router.post("/admin/reports/{report_id}/resolve", dependencies=[Depends(require_admin)])
async def resolve_content_report(request: Request, report_id: int, body: ResolveReportRequest):
    admin_id = request.state.user["id"]
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE content_reports SET reviewed_at = now(), reviewed_by = $2::uuid, resolution = $3 WHERE id = $1 RETURNING id",
            report_id, admin_id, body.resolution.strip(),
        )
    if row is None:
        raise HTTPException(404, "Report not found.")
    return {"ok": True}


@router.get("/admin/reports/review-time-stats", dependencies=[Depends(require_admin)])
async def content_report_review_time_stats():
    """SAF-2's "review time is tracked and reported monthly" -- this
    MEASURES the real average time-to-review from actual reports, it
    doesn't enforce the 24h target. An admin still has to act on the
    queue above; this is visibility, not automation."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT date_trunc('month', created_at) AS month,
                   count(*) AS reviewed_count,
                   avg(extract(epoch FROM reviewed_at - created_at) / 3600.0) AS avg_review_hours
            FROM content_reports WHERE reviewed_at IS NOT NULL
            GROUP BY month ORDER BY month DESC
            """
        )
    return {
        "months": [
            {"month": r["month"].date().isoformat(), "reviewed_count": r["reviewed_count"],
             "avg_review_hours": round(r["avg_review_hours"], 1) if r["avg_review_hours"] is not None else None}
            for r in rows
        ]
    }


# SAF-3: admin review queue for the automated mention-spike / duplicate-post
# scan (services/community_safety_service.py). Flags are held for review,
# never auto-removed -- resolving just marks them reviewed.

@router.get("/admin/flagged-content", dependencies=[Depends(require_admin)])
async def list_flagged_content():
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM flagged_content WHERE reviewed_at IS NULL ORDER BY created_at DESC")
    return {"flags": [_record_to_dict(r) for r in rows]}


@router.post("/admin/flagged-content/{flag_id}/resolve", dependencies=[Depends(require_admin)])
async def resolve_flagged_content(request: Request, flag_id: int):
    admin_id = request.state.user["id"]
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE flagged_content SET reviewed_at = now(), reviewed_by = $2::uuid WHERE id = $1 RETURNING id",
            flag_id, admin_id,
        )
    if row is None:
        raise HTTPException(404, "Flag not found.")
    return {"ok": True}


class CommentCreateRequest(BaseModel):
    body: str


@router.post("/posts/{post_id}/comments")
@limiter.limit("30/minute")
async def create_comment(request: Request, post_id: int, body: CommentCreateRequest):
    user_id = request.state.user["id"]
    text = body.body.strip()
    if not text or len(text) > MAX_COMMENT_BODY:
        raise HTTPException(422, f"body must be 1-{MAX_COMMENT_BODY} characters")
    async with service_conn() as conn:
        post = await conn.fetchrow("SELECT author_user_id FROM posts WHERE id = $1", post_id)
        if post is None:
            raise HTTPException(404, "Post not found.")
        display_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", user_id)
        record = await conn.fetchrow(
            "INSERT INTO post_comments (post_id, author_user_id, body) VALUES ($1, $2::uuid, $3) RETURNING *",
            post_id, user_id, text,
        )
        await _dispatch_mentions(conn, text, user_id, display_name or "Someone", post_id)
        if str(post["author_user_id"]) != user_id:
            summary = f"{display_name or 'Someone'} replied to your post"
            await conn.execute(
                """
                INSERT INTO social_notifications (user_id, notification_type, actor_user_id, post_id, summary)
                VALUES ($1::uuid, 'post_reply', $2::uuid, $3, $4)
                """,
                post["author_user_id"], user_id, post_id, summary,
            )
    if str(post["author_user_id"]) != user_id:
        await dispatch_alert(str(post["author_user_id"]), None, "post_reply", summary, f"{display_name or 'Someone'} replied to your post.")
    return {**_record_to_dict(record), "display_name": display_name}


@router.get("/posts/{post_id}/comments")
async def list_comments(post_id: int):
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.*, u.display_name FROM post_comments c LEFT JOIN users u ON u.id = c.author_user_id
            WHERE post_id = $1 ORDER BY c.is_accepted DESC, c.created_at
            """,
            post_id,
        )
    return {"comments": [_record_to_dict(r) for r in rows]}


# ---------------------------------------------------------------- BEG-4: Q&A (accepted answers)

@router.post("/comments/{comment_id}/accept")
@limiter.limit("30/minute")
async def accept_comment(request: Request, comment_id: int):
    """Only the asker (the post's author) can mark a comment as the
    accepted answer, and only one comment per post can hold it at a time
    -- enforced here (unset the old one, set the new one, in one
    transaction) and backed by a partial unique index at the DB level
    (post_comments_one_accepted_per_post) so a race can't leave two."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        comment = await conn.fetchrow("SELECT id, post_id FROM post_comments WHERE id = $1", comment_id)
        if comment is None:
            raise HTTPException(404, "Comment not found.")
        post = await conn.fetchrow("SELECT author_user_id FROM posts WHERE id = $1", comment["post_id"])
        if post is None or str(post["author_user_id"]) != user_id:
            raise HTTPException(403, "Only the person who asked the question can accept an answer.")
        async with conn.transaction():
            await conn.execute(
                "UPDATE post_comments SET is_accepted = false WHERE post_id = $1 AND is_accepted",
                comment["post_id"],
            )
            record = await conn.fetchrow(
                "UPDATE post_comments SET is_accepted = true WHERE id = $1 RETURNING *", comment_id
            )
    return _record_to_dict(record)


@router.post("/comments/{comment_id}/unaccept")
async def unaccept_comment(request: Request, comment_id: int):
    """Lets the asker change their mind without picking a different answer first."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        comment = await conn.fetchrow("SELECT id, post_id FROM post_comments WHERE id = $1", comment_id)
        if comment is None:
            raise HTTPException(404, "Comment not found.")
        post = await conn.fetchrow("SELECT author_user_id FROM posts WHERE id = $1", comment["post_id"])
        if post is None or str(post["author_user_id"]) != user_id:
            raise HTTPException(403, "Only the person who asked the question can do this.")
        record = await conn.fetchrow(
            "UPDATE post_comments SET is_accepted = false WHERE id = $1 RETURNING *", comment_id
        )
    return _record_to_dict(record)


# ---------------------------------------------------------------- SOC-6: groups

class GroupCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    topic: Optional[str] = None
    ticker: Optional[str] = None
    is_private: bool = False


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")[:60]


@router.post("/groups")
@limiter.limit("10/minute")
async def create_group(request: Request, body: GroupCreateRequest):
    user_id = request.state.user["id"]
    name = body.name.strip()
    if not (2 <= len(name) <= 80):
        raise HTTPException(422, "name must be 2-80 characters")
    slug = _slugify(name)
    ticker = _ticker(body.ticker) if body.ticker else None
    async with service_conn() as conn:
        try:
            record = await conn.fetchrow(
                """
                INSERT INTO groups (name, slug, description, topic, ticker, is_private, created_by_user_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7::uuid) RETURNING *
                """,
                name, slug, body.description, body.topic, ticker, body.is_private, user_id,
            )
        except Exception:
            raise HTTPException(409, "A group with that name already exists.")
        await conn.execute(
            "INSERT INTO group_members (group_id, user_id, role) VALUES ($1, $2::uuid, 'owner')",
            record["id"], user_id,
        )
    return _record_to_dict(record)


@router.get("/groups")
async def list_groups(request: Request):
    """Private groups are listed (so they're discoverable/joinable by
    name) but their posts stay gated -- see get_group/list_posts."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT g.*, (
                SELECT count(*) FROM group_members gm WHERE gm.group_id = g.id
            ) AS member_count,
            EXISTS(SELECT 1 FROM group_members gm WHERE gm.group_id = g.id AND gm.user_id = $1::uuid) AS is_member
            FROM groups g ORDER BY g.created_at DESC
            """,
            user_id,
        )
    return {"groups": [_record_to_dict(r) for r in rows]}


async def _require_group_access(conn, group_id: int, user_id: str) -> dict:
    group = await conn.fetchrow("SELECT * FROM groups WHERE id = $1", group_id)
    if group is None:
        raise HTTPException(404, "Group not found.")
    if group["is_private"]:
        is_member = await conn.fetchval(
            "SELECT 1 FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", group_id, user_id
        )
        if not is_member:
            raise HTTPException(403, "This is a private group -- join to see its content.")
    return dict(group)


@router.get("/groups/{group_id}")
async def get_group(request: Request, group_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        group = await _require_group_access(conn, group_id, user_id)
        members = await conn.fetch(
            """
            SELECT gm.user_id, gm.role, u.display_name FROM group_members gm
            LEFT JOIN users u ON u.id = gm.user_id WHERE gm.group_id = $1 ORDER BY gm.joined_at
            """,
            group_id,
        )
    return {**group, "members": [_record_to_dict(m) for m in members]}


@router.post("/groups/{group_id}/join")
async def join_group(request: Request, group_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        group = await conn.fetchrow("SELECT is_private FROM groups WHERE id = $1", group_id)
        if group is None:
            raise HTTPException(404, "Group not found.")
        if group["is_private"]:
            raise HTTPException(403, "This group is private -- ask a moderator to add you.")
        await conn.execute(
            "INSERT INTO group_members (group_id, user_id) VALUES ($1, $2::uuid) ON CONFLICT DO NOTHING",
            group_id, user_id,
        )
    return {"ok": True}


@router.post("/groups/{group_id}/leave")
async def leave_group(request: Request, group_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        role = await conn.fetchval(
            "SELECT role FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", group_id, user_id
        )
        if role == "owner":
            raise HTTPException(422, "The owner can't leave -- delete the group's role another way first.")
        await conn.execute("DELETE FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", group_id, user_id)
    return {"ok": True}


async def _require_moderator(conn, group_id: int, user_id: str) -> None:
    role = await conn.fetchval(
        "SELECT role FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", group_id, user_id
    )
    if role not in ("moderator", "owner"):
        raise HTTPException(403, "Only a moderator or the owner can do this.")


@router.post("/groups/{group_id}/members/{member_id}/remove")
async def remove_member(request: Request, group_id: int, member_id: str):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_moderator(conn, group_id, user_id)
        await conn.execute(
            "DELETE FROM group_members WHERE group_id = $1 AND user_id = $2::uuid AND role <> 'owner'",
            group_id, member_id,
        )
    return {"ok": True}


@router.delete("/groups/{group_id}/posts/{post_id}")
async def remove_group_post(request: Request, group_id: int, post_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_moderator(conn, group_id, user_id)
        row = await conn.fetchrow(
            "UPDATE posts SET hidden = true WHERE id = $1 AND group_id = $2 RETURNING id", post_id, group_id
        )
    if row is None:
        raise HTTPException(404, "Post not found in this group.")
    return {"ok": True}


# ---------------------------------------------------------------- BEG-5: group sessions

class SessionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    scheduled_at: str  # ISO datetime, parsed by asyncpg/timestamptz


@router.post("/groups/{group_id}/sessions")
@limiter.limit("10/minute")
async def create_session(request: Request, group_id: int, body: SessionCreateRequest):
    """Only a mentor (verified_badge + enough accepted answers, see
    web/backend/social_badges.py::recompute_mentor_badges) who is also a
    member of this group can host a session here."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        is_mentor = await conn.fetchval("SELECT mentor_badge FROM users WHERE id = $1::uuid", user_id)
        if not is_mentor:
            raise HTTPException(403, "Only mentors can host a group session.")
        is_member = await conn.fetchval(
            "SELECT 1 FROM group_members WHERE group_id = $1 AND user_id = $2::uuid", group_id, user_id
        )
        if not is_member:
            raise HTTPException(403, "You must be a member of this group to host a session in it.")
        record = await conn.fetchrow(
            """
            INSERT INTO group_sessions (group_id, host_user_id, title, description, scheduled_at)
            VALUES ($1, $2::uuid, $3, $4, $5::timestamptz)
            RETURNING *
            """,
            group_id, user_id, body.title.strip(), body.description, body.scheduled_at,
        )
    return _record_to_dict(record)


@router.get("/groups/{group_id}/sessions")
async def list_sessions(request: Request, group_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_group_access(conn, group_id, user_id)
        rows = await conn.fetch(
            """
            SELECT s.*, u.display_name AS host_display_name FROM group_sessions s
            LEFT JOIN users u ON u.id = s.host_user_id
            WHERE s.group_id = $1 ORDER BY s.scheduled_at
            """,
            group_id,
        )
    return {"sessions": [_record_to_dict(r) for r in rows]}


# ---------------------------------------------------------------- SOC-7: chat rooms

class ChatMessageCreateRequest(BaseModel):
    body: str


@router.get("/chat/{room}/messages")
async def list_chat_messages(room: str, since_id: int = 0, limit: int = Query(100, le=200)):
    try:
        room = validate_room(room)
    except ValueError as e:
        raise HTTPException(422, str(e))
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT cm.*, u.display_name FROM chat_messages cm LEFT JOIN users u ON u.id = cm.user_id
            WHERE room = $1 AND cm.id > $2 ORDER BY cm.id LIMIT $3
            """,
            room, since_id, limit,
        )
    return {"room": room, "messages": [_record_to_dict(r) for r in rows], "market_open": is_market_hours_now()}


@router.post("/chat/{room}/messages")
@limiter.limit("30/minute")
async def post_chat_message(request: Request, room: str, body: ChatMessageCreateRequest):
    user_id = request.state.user["id"]
    try:
        room = validate_room(room)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not is_market_hours_now():
        raise HTTPException(422, "Chat is only open during market hours (Mon-Fri, 9:30-16:00 ET).")
    text = body.body.strip()
    if not text or len(text) > MAX_CHAT_BODY:
        raise HTTPException(422, f"body must be 1-{MAX_CHAT_BODY} characters")
    async with service_conn() as conn:
        record = await conn.fetchrow(
            "INSERT INTO chat_messages (room, user_id, body) VALUES ($1, $2::uuid, $3) RETURNING *",
            room, user_id, text,
        )
        display_name = await conn.fetchval("SELECT display_name FROM users WHERE id = $1::uuid", user_id)
    return {**_record_to_dict(record), "display_name": display_name}


# ---------------------------------------------------------------- SOC-7: direct messages

class DmSendRequest(BaseModel):
    body: str


async def _dm_allowed(conn, sender_id: str, recipient_id: str) -> bool:
    """"Only from people the user follows or allows" -- allowed if the
    RECIPIENT follows the sender, or has explicitly allow-listed them.
    SAF-7: a block always wins, in either direction -- a sender the
    recipient has blocked can't DM them even if previously followed/
    allow-listed, and blocking someone also stops you from DMing them."""
    blocked = await conn.fetchval(
        """
        SELECT 1 FROM user_blocks
        WHERE (blocker_user_id = $1::uuid AND blocked_user_id = $2::uuid)
           OR (blocker_user_id = $2::uuid AND blocked_user_id = $1::uuid)
        """,
        recipient_id, sender_id,
    )
    if blocked:
        return False
    follows = await conn.fetchval(
        "SELECT 1 FROM author_follows WHERE follower_user_id = $1::uuid AND followed_user_id = $2::uuid",
        recipient_id, sender_id,
    )
    if follows:
        return True
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM dm_allowed_senders WHERE user_id = $1::uuid AND allowed_sender_user_id = $2::uuid",
            recipient_id, sender_id,
        )
    )


@router.get("/messages")
async def list_conversations(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (other_id) other_id, body, created_at FROM (
                SELECT CASE WHEN sender_user_id = $1::uuid THEN recipient_user_id ELSE sender_user_id END AS other_id,
                       body, created_at
                FROM direct_messages WHERE sender_user_id = $1::uuid OR recipient_user_id = $1::uuid
            ) t ORDER BY other_id, created_at DESC
            """,
            user_id,
        )
        others = [r["other_id"] for r in rows]
        names = await conn.fetch("SELECT id, display_name FROM users WHERE id = ANY($1::uuid[])", others) if others else []
    name_by_id = {str(n["id"]): n["display_name"] for n in names}
    return {
        "conversations": [
            {"other_user_id": str(r["other_id"]), "display_name": name_by_id.get(str(r["other_id"])), "last_message": r["body"], "last_at": r["created_at"]}
            for r in rows
        ]
    }


@router.get("/messages/{other_user_id}")
async def get_thread(request: Request, other_user_id: str, limit: int = Query(100, le=200)):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM direct_messages
            WHERE (sender_user_id = $1::uuid AND recipient_user_id = $2::uuid)
               OR (sender_user_id = $2::uuid AND recipient_user_id = $1::uuid)
            ORDER BY created_at DESC LIMIT $3
            """,
            user_id, other_user_id, limit,
        )
    return {"messages": [_record_to_dict(r) for r in reversed(rows)]}


@router.post("/messages/{other_user_id}")
@limiter.limit("30/minute")
async def send_dm(request: Request, other_user_id: str, body: DmSendRequest):
    user_id = request.state.user["id"]
    if user_id == other_user_id:
        raise HTTPException(422, "You can't message yourself.")
    text = body.body.strip()
    if not text or len(text) > MAX_COMMENT_BODY:
        raise HTTPException(422, f"body must be 1-{MAX_COMMENT_BODY} characters")
    async with service_conn() as conn:
        if not await _dm_allowed(conn, user_id, other_user_id):
            raise HTTPException(403, "This person only accepts messages from people they follow or allow.")
    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            "INSERT INTO direct_messages (sender_user_id, recipient_user_id, body) VALUES ($1::uuid, $2::uuid, $3) RETURNING *",
            user_id, other_user_id, text,
        )
    return _record_to_dict(record)


@router.post("/dm-allowed/{sender_id}")
async def allow_dm_sender(request: Request, sender_id: str):
    user_id = request.state.user["id"]
    if user_id == sender_id:
        raise HTTPException(422, "You don't need to allow yourself.")
    async with user_conn(user_id) as conn:
        await conn.execute(
            "INSERT INTO dm_allowed_senders (user_id, allowed_sender_user_id) VALUES ($1::uuid, $2::uuid) ON CONFLICT DO NOTHING",
            user_id, sender_id,
        )
    return {"ok": True}


@router.delete("/dm-allowed/{sender_id}")
async def revoke_dm_sender(request: Request, sender_id: str):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        await conn.execute(
            "DELETE FROM dm_allowed_senders WHERE user_id = $1::uuid AND allowed_sender_user_id = $2::uuid",
            user_id, sender_id,
        )
    return {"ok": True}
