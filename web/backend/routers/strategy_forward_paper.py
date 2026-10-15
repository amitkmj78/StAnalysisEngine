"""STB-6: link a saved strategy to its own real Alpaca paper account,
accept the disclosure, enable/disable/kill the daily forward-paper job
(services/strategy_forward_paper.py), and read its status/track record.

Deliberately its own file, not folded into strategy_builder.py -- that
module's own docstring explicitly deferred "writing into a paper account
unattended" out of scope for STS-5; this is the first time that line gets
crossed, scoped to a strategy's own account (never a follower's, which
stays deferred).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import asyncpg
from services import alpaca_trading_client
from services.alpaca_trading_client import AlpacaTradingError
from services.strategy_forward_paper import get_track_record
from web.backend.auth import verify_bearer_token
from web.backend.crypto_utils import encrypt_token
from web.backend.db import user_conn

router = APIRouter(
    prefix="/api/v1/strategy-builder/saved",
    tags=["strategy-forward-paper"],
    dependencies=[Depends(verify_bearer_token)],
)

_ACCOUNT_PUBLIC_COLUMNS = (
    "id, saved_strategy_id, api_key_id, alpaca_account_id, account_number, status, enabled, kill_engaged, "
    "disclosure_accepted_at, last_run_at, last_run_status, created_at"
)


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


async def _get_account_or_404(conn, saved_id: int) -> dict:
    row = await conn.fetchrow(
        f"SELECT {_ACCOUNT_PUBLIC_COLUMNS} FROM strategy_paper_accounts WHERE saved_strategy_id = $1", saved_id
    )
    if row is None:
        raise HTTPException(404, "No forward-paper account linked yet for this strategy.")
    return dict(row)


async def _assert_strategy_owned(conn, saved_id: int) -> None:
    # RLS already scopes this to the caller's own rows -- a hit confirms
    # both existence and ownership in one query.
    row = await conn.fetchval("SELECT id FROM saved_strategies WHERE id = $1", saved_id)
    if row is None:
        raise HTTPException(404, "Saved strategy not found.")


class LinkRequest(BaseModel):
    api_key_id: str = Field(min_length=1)
    api_secret_key: str = Field(min_length=1)


@router.post("/{saved_id}/forward-paper/link")
async def link_forward_paper(request: Request, saved_id: int, body: LinkRequest):
    user_id = request.state.user["id"]

    # Ownership first -- cheap and local -- before the Alpaca round-trip
    # and encryption work, so an invalid saved_id fails fast.
    async with user_conn(user_id) as conn:
        await _assert_strategy_owned(conn, saved_id)

    try:
        account_info = await run_in_threadpool(alpaca_trading_client.get_account, body.api_key_id, body.api_secret_key)
    except AlpacaTradingError as e:
        raise HTTPException(422, f"Alpaca rejected this key pair: {e.message}")

    secret_encrypted = encrypt_token(body.api_secret_key)
    async with user_conn(user_id) as conn:
        try:
            row = await conn.fetchrow(
                f"""
                INSERT INTO strategy_paper_accounts (
                    saved_strategy_id, user_id, api_key_id, api_secret_key_encrypted,
                    alpaca_account_id, account_number, status
                ) VALUES ($1, $2::uuid, $3, $4, $5, $6, 'active')
                RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
                """,
                saved_id, user_id, body.api_key_id, secret_encrypted,
                account_info.get("id"), account_info.get("account_number"),
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(409, "This strategy already has a forward-paper account linked. Unlink it first to link a different one.")
    return {"account": _record_to_dict(row)}


@router.post("/{saved_id}/forward-paper/disclosure-accept")
async def accept_forward_paper_disclosure(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE strategy_paper_accounts SET disclosure_accepted_at = now()
            WHERE saved_strategy_id = $1
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            saved_id,
        )
    if row is None:
        raise HTTPException(404, "No forward-paper account linked yet for this strategy.")
    return {"account": _record_to_dict(row)}


@router.post("/{saved_id}/forward-paper/enable")
async def enable_forward_paper(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        account = await _get_account_or_404(conn, saved_id)
        if account["disclosure_accepted_at"] is None:
            raise HTTPException(403, "Accept the forward-paper-trading disclosure before enabling it.")
        row = await conn.fetchrow(
            f"""
            UPDATE strategy_paper_accounts SET enabled = true WHERE saved_strategy_id = $1
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            saved_id,
        )
    return {"account": _record_to_dict(row)}


@router.post("/{saved_id}/forward-paper/disable")
async def disable_forward_paper(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE strategy_paper_accounts SET enabled = false WHERE saved_strategy_id = $1
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            saved_id,
        )
    if row is None:
        raise HTTPException(404, "No forward-paper account linked yet for this strategy.")
    return {"account": _record_to_dict(row)}


class KillSwitchRequest(BaseModel):
    engaged: bool


@router.post("/{saved_id}/forward-paper/kill-switch")
async def set_forward_paper_kill_switch(request: Request, saved_id: int, body: KillSwitchRequest):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE strategy_paper_accounts SET kill_engaged = $2 WHERE saved_strategy_id = $1
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            saved_id, body.engaged,
        )
    if row is None:
        raise HTTPException(404, "No forward-paper account linked yet for this strategy.")
    return {"account": _record_to_dict(row)}


@router.get("/{saved_id}/forward-paper/status")
async def get_forward_paper_status(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        account = await _get_account_or_404(conn, saved_id)
        positions = await conn.fetch(
            "SELECT ticker, qty, entry_price, entry_atr, peak_price, entry_date, days_held, cooldown_until_date "
            "FROM strategy_paper_positions WHERE strategy_paper_account_id = $1 AND qty > 0",
            account["id"],
        )
        events = await conn.fetch(
            "SELECT * FROM strategy_paper_order_events WHERE strategy_paper_account_id = $1 "
            "ORDER BY created_at DESC LIMIT 50",
            account["id"],
        )
        runs = await conn.fetch(
            "SELECT id, status, equity, reason, created_at FROM strategy_paper_runs "
            "WHERE strategy_paper_account_id = $1 ORDER BY created_at DESC LIMIT 10",
            account["id"],
        )
    return {
        "account": account,
        "positions": [_record_to_dict(r) for r in positions],
        "journal": [_record_to_dict(r) for r in events],
        "runs": [_record_to_dict(r) for r in runs],
    }


@router.delete("/{saved_id}/forward-paper/link")
async def unlink_forward_paper(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchval(
            "DELETE FROM strategy_paper_accounts WHERE saved_strategy_id = $1 RETURNING id", saved_id
        )
    if row is None:
        raise HTTPException(404, "No forward-paper account linked yet for this strategy.")
    return {"ok": True}


@router.get("/{saved_id}/forward-paper/track-record")
async def get_forward_paper_track_record(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    return await get_track_record(saved_id, user_id)
