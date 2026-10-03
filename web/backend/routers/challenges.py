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
from pydantic import BaseModel, Field

from services.challenge_service import compute_member_performance, generate_join_code
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


class JoinChallengeRequest(BaseModel):
    join_code: str = Field(min_length=1, max_length=20)


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
                INSERT INTO challenges (name, created_by, join_code, start_date, end_date)
                VALUES ($1, $2::uuid, $3, $4, $5) RETURNING id
                """,
                body.name, user_id, join_code, start_date, end_date,
            )
            await conn.execute(
                "INSERT INTO challenge_members (challenge_id, user_id) VALUES ($1, $2::uuid)",
                challenge_id, user_id,
            )

    return {
        "id": challenge_id, "name": body.name, "join_code": join_code,
        "start_date": str(start_date), "end_date": str(end_date),
    }


@router.post("/join")
@limiter.limit("10/minute")
async def join_challenge(request: Request, body: JoinChallengeRequest):
    await enforce_daily_quota(request, "challenges/join")
    user_id = request.state.user["id"]

    async with service_conn() as conn:
        await _require_paper_account(conn, user_id)

        challenge = await conn.fetchrow(
            "SELECT id, name, start_date, end_date FROM challenges WHERE join_code = $1", body.join_code.upper()
        )
        if challenge is None:
            raise HTTPException(404, "No challenge found for that code.")
        if challenge["end_date"] < date.today():
            raise HTTPException(400, "This challenge already ended.")

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


@router.get("")
async def list_my_challenges(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.name, c.start_date, c.end_date,
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
        "members": [r["email"] for r in member_rows],
    }


@router.get("/{challenge_id}/leaderboard")
async def get_challenge_leaderboard(request: Request, challenge_id: int):
    """Returns return_pct/max_drawdown_pct/annualized_volatility_pct/
    days_of_data per member, sorted by return_pct descending (members
    with no computable return sort last, not hidden -- see
    challenge_service.compute_member_performance's own docstring on why
    thin data is disclosed rather than faked or dropped). Never returns
    a member's raw equity -- only the percentages compute_member_
    performance derives from it."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await _require_membership(conn, challenge_id, user_id)
        challenge = await conn.fetchrow("SELECT start_date, end_date FROM challenges WHERE id = $1", challenge_id)
        if challenge is None:
            raise HTTPException(404, "Challenge not found.")
        start_date, end_date = challenge["start_date"], min(challenge["end_date"], date.today())

        member_rows = await conn.fetch(
            """
            SELECT m.user_id, u.email, a.id AS alpaca_paper_account_id
            FROM challenge_members m
            JOIN users u ON u.id = m.user_id
            LEFT JOIN alpaca_paper_accounts a ON a.user_id = m.user_id
            WHERE m.challenge_id = $1
            """,
            challenge_id,
        )

        entries = []
        for member in member_rows:
            if member["alpaca_paper_account_id"] is None:
                entries.append({
                    "email": member["email"], "return_pct": None, "max_drawdown_pct": None,
                    "annualized_volatility_pct": None, "days_of_data": 0,
                })
                continue

            snapshot_rows = await conn.fetch(
                """
                SELECT as_of_date, equity FROM paper_account_equity_snapshots
                WHERE alpaca_paper_account_id = $1 AND as_of_date BETWEEN $2 AND $3
                """,
                member["alpaca_paper_account_id"], start_date, end_date,
            )
            performance = compute_member_performance(
                [{"as_of_date": r["as_of_date"], "equity": r["equity"]} for r in snapshot_rows],
                start_date, end_date,
            )
            entries.append({"email": member["email"], **performance})

    entries.sort(key=lambda e: (e["return_pct"] is None, -(e["return_pct"] or 0)))
    return {"start_date": str(start_date), "end_date": str(end_date), "entries": entries}


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
