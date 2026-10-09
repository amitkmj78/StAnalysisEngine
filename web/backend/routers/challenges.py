"""
PPR-2: monthly paper-trading challenges with a friend leaderboard.

"Friends" here means a private challenge joined via a shareable code --
this app has no existing social graph to build on. Every endpoint goes
through service_conn() rather than user_conn(), since a leaderboard
needs to read other members' rows; membership/ownership is an explicit
check in each endpoint instead of an RLS policy (challenges/
challenge_members carry no RLS at all -- see aws_deploy.py's comment on
those tables). Raw paper-account equity never leaves this router --
only the derived percentages computed by challenge_service.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from starlette.concurrency import run_in_threadpool

from services.challenge_leaderboard import build_leaderboard
from services.challenge_service import (
    mask_email,
    DEFAULT_SCORING,
    SCORING_METHODS,
    rebase_to_100,
    compute_member_performance,
    generate_join_code,
    score_for,
)
from services.email_service import APP_URL, send_alert_email
from services.quant_model_service import MODEL_MEMBER_LABEL, model_snapshots
from services.yfinance_cache import get_cached_history
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/challenges", tags=["challenges"], dependencies=[Depends(verify_bearer_token)])




def _current_month_bounds() -> tuple[date, date]:
    today = date.today()
    start = date(today.year, today.month, 1)
    end = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    return start, end


async def _require_paper_account(conn, user_id: str) -> None:
    row = await conn.fetchrow("SELECT id FROM alpaca_paper_accounts WHERE user_id = $1::uuid", user_id)
    if row is None:
        raise HTTPException(
            400, "Link a paper-trading account first (Portfolio → Paper Trading) before joining a challenge."
        )


async def _require_membership(conn, challenge_id: int, user_id: str) -> None:
    row = await conn.fetchrow(
        "SELECT 1 FROM challenge_members WHERE challenge_id = $1 AND user_id = $2::uuid", challenge_id, user_id
    )
    if row is None:
        raise HTTPException(403, "You're not a member of this challenge.")


class CreateChallengeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    scoring: str = DEFAULT_SCORING
    include_quant_model: bool = False
    # BEG-6: forces 'diversified' scoring -- "not raw gains" is enforced
    # on create, not just suggested in the UI.
    beginner_only: bool = False


class JoinChallengeRequest(BaseModel):
    join_code: str = Field(min_length=1, max_length=20)


class DiscoverabilityRequest(BaseModel):
    discoverable: bool


class InviteUserRequest(BaseModel):
    user_id: str


class InviteEmailRequest(BaseModel):
    email: EmailStr


def _invite_email_body(challenge_name: str, join_code: str, inviter_email: str) -> tuple[str, str]:
    subject = f"{inviter_email} invited you to the \"{challenge_name}\" challenge"
    body = (
        f"{inviter_email} invited you to join their paper-trading challenge, \"{challenge_name}\", "
        f"on StAnalysisEngine.\n\n"
        f"Join code: {join_code}\n"
        f"Join here: {APP_URL}/challenges?join={join_code}\n\n"
        f"You'll need a linked paper-trading account (free, simulated money) to show up on the leaderboard --"
        f" the join page has a link to set one up if you don't have one yet."
    )
    return subject, body


@router.post("")
@limiter.limit("10/minute")
async def create_challenge(request: Request, body: CreateChallengeRequest):
    await enforce_daily_quota(request, "challenges/create")
    user_id = request.state.user["id"]

    default_start, default_end = _current_month_bounds()
    start_date = body.start_date or default_start
    end_date = body.end_date or default_end
    if end_date <= start_date:
        raise HTTPException(400, "end_date must be after start_date.")
    if body.scoring not in SCORING_METHODS:
        raise HTTPException(400, f"scoring must be one of: {', '.join(SCORING_METHODS)}.")
    if body.beginner_only and body.scoring != "diversified":
        raise HTTPException(422, "A beginner-only challenge must use the 'diversified' scoring method.")

    async with service_conn() as conn:
        await _require_paper_account(conn, user_id)

        join_code = generate_join_code()
        for _ in range(5):
            existing = await conn.fetchval("SELECT 1 FROM challenges WHERE join_code = $1", join_code)
            if existing is None:
                break
            join_code = generate_join_code()

        async with conn.transaction():
            challenge_id = await conn.fetchval(
                """
                INSERT INTO challenges (name, created_by, join_code, start_date, end_date, scoring, include_quant_model, beginner_only)
                VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8) RETURNING id
                """,
                body.name, user_id, join_code, start_date, end_date, body.scoring, body.include_quant_model,
                body.beginner_only,
            )
            await conn.execute(
                "INSERT INTO challenge_members (challenge_id, user_id) VALUES ($1, $2::uuid)",
                challenge_id, user_id,
            )

    return {
        "id": challenge_id, "name": body.name, "join_code": join_code,
        "start_date": str(start_date), "end_date": str(end_date), "beginner_only": body.beginner_only,
    }


@router.post("/join")
@limiter.limit("10/minute")
async def join_challenge(request: Request, body: JoinChallengeRequest):
    await enforce_daily_quota(request, "challenges/join")
    user_id = request.state.user["id"]

    async with service_conn() as conn:
        await _require_paper_account(conn, user_id)

        challenge = await conn.fetchrow(
            "SELECT id, name, start_date, end_date, beginner_only FROM challenges WHERE join_code = $1",
            body.join_code.upper(),
        )
        if challenge is None:
            raise HTTPException(404, "No challenge found for that code.")
        if challenge["end_date"] < date.today():
            raise HTTPException(400, "This challenge already ended.")
        if challenge["beginner_only"]:
            level = await conn.fetchval("SELECT experience_level FROM users WHERE id = $1::uuid", user_id)
            if level != "beginner":
                raise HTTPException(403, "This challenge is only open to members in Beginner mode (see Settings).")

        already = await conn.fetchval(
            "SELECT 1 FROM challenge_members WHERE challenge_id = $1 AND user_id = $2::uuid",
            challenge["id"], user_id,
        )
        if already is not None:
            raise HTTPException(409, "You're already in this challenge.")

        await conn.execute(
            "INSERT INTO challenge_members (challenge_id, user_id) VALUES ($1, $2::uuid)",
            challenge["id"], user_id,
        )

    return {
        "id": challenge["id"], "name": challenge["name"],
        "start_date": str(challenge["start_date"]), "end_date": str(challenge["end_date"]),
    }


@router.get("/discoverability")
async def get_discoverability(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        value = await conn.fetchval(
            "SELECT discoverable_for_challenges FROM users WHERE id = $1::uuid", user_id
        )
    return {"discoverable": bool(value)}


@router.post("/discoverability")
@limiter.limit("10/minute")
async def set_discoverability(request: Request, body: DiscoverabilityRequest):
    await enforce_daily_quota(request, "challenges/discoverability")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE users SET discoverable_for_challenges = $1 WHERE id = $2::uuid", body.discoverable, user_id
        )
    return {"discoverable": body.discoverable}


@router.get("/invites")
async def list_my_invites(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT i.id, i.challenge_id, c.name AS challenge_name, u.email AS invited_by_email, i.created_at
            FROM challenge_invites i
            JOIN challenges c ON c.id = i.challenge_id
            JOIN users u ON u.id = i.invited_by
            WHERE i.invited_user_id = $1::uuid AND i.status = 'pending'
            ORDER BY i.created_at DESC
            """,
            user_id,
        )
    return {
        "invites": [
            {
                "id": r["id"], "challenge_id": r["challenge_id"], "challenge_name": r["challenge_name"],
                "invited_by_label": mask_email(r["invited_by_email"]), "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ]
    }


@router.post("/invites/{invite_id}/accept")
@limiter.limit("10/minute")
async def accept_invite(request: Request, invite_id: int):
    await enforce_daily_quota(request, "challenges/invites/accept")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        invite = await conn.fetchrow(
            "SELECT challenge_id FROM challenge_invites WHERE id = $1 AND invited_user_id = $2::uuid AND status = 'pending'",
            invite_id, user_id,
        )
        if invite is None:
            raise HTTPException(404, "No pending invite found.")
        await _require_paper_account(conn, user_id)

        async with conn.transaction():
            await conn.execute(
                "UPDATE challenge_invites SET status = 'accepted', responded_at = now() WHERE id = $1", invite_id
            )
            await conn.execute(
                """
                INSERT INTO challenge_members (challenge_id, user_id) VALUES ($1, $2::uuid)
                ON CONFLICT (challenge_id, user_id) DO NOTHING
                """,
                invite["challenge_id"], user_id,
            )
    return {"ok": True, "challenge_id": invite["challenge_id"]}


@router.post("/invites/{invite_id}/decline")
@limiter.limit("10/minute")
async def decline_invite(request: Request, invite_id: int):
    await enforce_daily_quota(request, "challenges/invites/decline")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        result = await conn.execute(
            """
            UPDATE challenge_invites SET status = 'declined', responded_at = now()
            WHERE id = $1 AND invited_user_id = $2::uuid AND status = 'pending'
            """,
            invite_id, user_id,
        )
    if result == "UPDATE 0":
        raise HTTPException(404, "No pending invite found.")
    return {"ok": True}


@router.get("")
async def list_my_challenges(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.name, c.start_date, c.end_date, c.beginner_only,
                   (SELECT count(*) FROM challenge_members m2 WHERE m2.challenge_id = c.id) AS member_count
            FROM challenges c
            JOIN challenge_members m ON m.challenge_id = c.id
            WHERE m.user_id = $1::uuid
            ORDER BY c.start_date DESC
            """,
            user_id,
        )
    return {
        "challenges": [
            {
                "id": r["id"], "name": r["name"], "start_date": str(r["start_date"]),
                "end_date": str(r["end_date"]), "member_count": r["member_count"],
                "beginner_only": r["beginner_only"],
            }
            for r in rows
        ]
    }


@router.get("/{challenge_id}")
async def get_challenge(request: Request, challenge_id: int):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        challenge = await conn.fetchrow("SELECT * FROM challenges WHERE id = $1", challenge_id)
        if challenge is None:
            raise HTTPException(404, "Challenge not found.")
        member_rows = await conn.fetch(
            """
            SELECT u.email FROM challenge_members m JOIN users u ON u.id = m.user_id
            WHERE m.challenge_id = $1 ORDER BY m.joined_at
            """,
            challenge_id,
        )

    return {
        "id": challenge["id"], "name": challenge["name"], "join_code": challenge["join_code"],
        "start_date": str(challenge["start_date"]), "end_date": str(challenge["end_date"]),
        "scoring": challenge["scoring"], "scoring_label": SCORING_METHODS[challenge["scoring"]],
        "members": [mask_email(r["email"]) for r in member_rows],
    }


@router.get("/{challenge_id}/discoverable-users")
async def list_discoverable_users(request: Request, challenge_id: int):
    """"Connect with the community": every user who's opted in (default
    on -- see discoverable_for_challenges's docstring in aws_deploy.py),
    minus the caller, minus whoever's already a member or already has a
    pending invite for this specific challenge."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        rows = await conn.fetch(
            """
            SELECT u.id, u.email FROM users u
            WHERE u.discoverable_for_challenges AND u.id != $1::uuid
              AND u.id NOT IN (SELECT user_id FROM challenge_members WHERE challenge_id = $2)
              AND u.id NOT IN (
                SELECT invited_user_id FROM challenge_invites WHERE challenge_id = $2 AND status = 'pending'
              )
            ORDER BY u.email
            """,
            user_id, challenge_id,
        )
    return {"users": [{"id": str(r["id"]), "label": mask_email(r["email"])} for r in rows]}


@router.post("/{challenge_id}/invite-user")
@limiter.limit("10/minute")
async def invite_user(request: Request, challenge_id: int, body: InviteUserRequest):
    await enforce_daily_quota(request, "challenges/invite-user")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        challenge = await conn.fetchrow("SELECT name, join_code FROM challenges WHERE id = $1", challenge_id)
        if challenge is None:
            raise HTTPException(404, "Challenge not found.")
        inviter = await conn.fetchrow("SELECT email FROM users WHERE id = $1::uuid", user_id)

        target = await conn.fetchrow(
            "SELECT email FROM users WHERE id = $1::uuid AND discoverable_for_challenges", body.user_id
        )
        if target is None:
            raise HTTPException(404, "That user isn't available to invite.")

        already_member = await conn.fetchval(
            "SELECT 1 FROM challenge_members WHERE challenge_id = $1 AND user_id = $2::uuid",
            challenge_id, body.user_id,
        )
        if already_member is not None:
            raise HTTPException(409, "That person is already in this challenge.")

        inserted = await conn.fetchval(
            """
            INSERT INTO challenge_invites (challenge_id, invited_user_id, invited_by)
            VALUES ($1, $2::uuid, $3::uuid)
            ON CONFLICT (challenge_id, invited_user_id) DO NOTHING RETURNING id
            """,
            challenge_id, body.user_id, user_id,
        )
        if inserted is None:
            raise HTTPException(409, "That person already has a pending invite to this challenge.")

    subject, text_body = _invite_email_body(challenge["name"], challenge["join_code"], inviter["email"])
    await run_in_threadpool(send_alert_email, target["email"], subject, text_body)
    return {"ok": True, "invited_label": mask_email(target["email"])}


@router.post("/{challenge_id}/invite-email")
@limiter.limit("10/minute")
async def invite_by_email(request: Request, challenge_id: int, body: InviteEmailRequest):
    """Email a join link/code to anyone, with or without an account. If
    the address matches an existing, discoverable user, this also
    creates the same pending request as invite-user above, so accepting
    just requires clicking the link in the email rather than re-finding
    the challenge by code -- a found/not-found account state isn't
    distinguished in the response, since this isn't a security-relevant
    enumeration surface, just a UX nicety."""
    await enforce_daily_quota(request, "challenges/invite-email")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        challenge = await conn.fetchrow("SELECT name, join_code FROM challenges WHERE id = $1", challenge_id)
        if challenge is None:
            raise HTTPException(404, "Challenge not found.")
        inviter = await conn.fetchrow("SELECT email FROM users WHERE id = $1::uuid", user_id)

        target = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 AND discoverable_for_challenges", body.email.lower()
        )
        if target is not None:
            await conn.execute(
                """
                INSERT INTO challenge_invites (challenge_id, invited_user_id, invited_by)
                VALUES ($1, $2::uuid, $3::uuid)
                ON CONFLICT (challenge_id, invited_user_id) DO NOTHING
                """,
                challenge_id, target["id"], user_id,
            )

    subject, text_body = _invite_email_body(challenge["name"], challenge["join_code"], inviter["email"])
    await run_in_threadpool(send_alert_email, body.email, subject, text_body)
    return {"ok": True}


@router.get("/{challenge_id}/leaderboard")
async def get_challenge_leaderboard(request: Request, challenge_id: int):
    """Per-member return, drawdown, volatility, score and badges. Members appear
    only by masked label, and raw equity never leaves the server."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
    board = await build_leaderboard(challenge_id)
    if board is None:
        raise HTTPException(404, "Challenge not found.")
    for entry in board["entries"]:
        entry.pop("user_id", None)
    return board


@router.get("/{challenge_id}/equity-curves")
async def get_challenge_equity_curves(request: Request, challenge_id: int):
    """Each member's account value rebased to 100 on the first snapshot of the
    window, with SPY rebased the same way over the same dates. Only these
    rebased values leave the server, never a raw equity figure."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        challenge = await conn.fetchrow(
            "SELECT start_date, end_date, include_quant_model FROM challenges WHERE id = $1", challenge_id
        )
        if challenge is None:
            raise HTTPException(404, "Challenge not found.")
        start_date, end_date = challenge["start_date"], min(challenge["end_date"], date.today())
        member_rows = await conn.fetch(
            """
            SELECT u.email, a.id AS alpaca_paper_account_id
            FROM challenge_members m JOIN users u ON u.id = m.user_id
            -- A user can now link one paper account per portfolio; a plain
            -- join would duplicate this member once per linked account, so
            -- this picks their oldest (primary) one deterministically, same
            -- choice the trading agent itself makes (services/agent/
            -- runner.py::_load_paper_account) -- challenges aren't
            -- portfolio-scoped, so one consistent account per member is
            -- the correct behavior here, not a stopgap.
            LEFT JOIN alpaca_paper_accounts a ON a.id = (
                SELECT id FROM alpaca_paper_accounts WHERE user_id = m.user_id ORDER BY created_at ASC LIMIT 1
            )
            WHERE m.challenge_id = $1
            """,
            challenge_id,
        )
        members = []
        for m in member_rows:
            points: list[dict] = []
            if m["alpaca_paper_account_id"] is not None:
                rows = await conn.fetch(
                    """
                    SELECT as_of_date, equity FROM paper_account_equity_snapshots
                    WHERE alpaca_paper_account_id = $1 AND as_of_date BETWEEN $2 AND $3
                    ORDER BY as_of_date
                    """,
                    m["alpaca_paper_account_id"], start_date, end_date,
                )
                points = rebase_to_100([(r["as_of_date"], r["equity"]) for r in rows])
            members.append({"member": mask_email(m["email"]), "points": points})

    if challenge["include_quant_model"]:
        model_snaps = await model_snapshots(start_date, end_date)
        members.append({"member": MODEL_MEMBER_LABEL, "points": rebase_to_100([(x["as_of_date"], x["equity"]) for x in model_snaps])})

    spy_points = await _spy_rebased_points(start_date, end_date)
    return {"start_date": str(start_date), "end_date": str(end_date), "spy": spy_points, "members": members}


async def _spy_rebased_points(start_date: date, end_date: date) -> list[dict]:
    try:
        closes = (await run_in_threadpool(get_cached_history, "SPY", "2y", True))["Close"]
    except Exception:  # noqa: BLE001
        return []
    window = closes.loc[str(start_date): str(end_date)].dropna()
    return rebase_to_100([(idx.date(), v) for idx, v in window.items()])


@router.delete("/{challenge_id}/leave")
@limiter.limit("10/minute")
async def leave_challenge(request: Request, challenge_id: int):
    await enforce_daily_quota(request, "challenges/leave")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        result = await conn.execute(
            "DELETE FROM challenge_members WHERE challenge_id = $1 AND user_id = $2::uuid", challenge_id, user_id
        )
    if result == "DELETE 0":
        raise HTTPException(404, "You're not a member of this challenge.")
    return {"ok": True}
