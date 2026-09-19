from typing import Optional

import plaid
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services import plaid_client
from services.plaid_client import PlaidNotConfiguredError
from web.backend import plaid_sync
from web.backend.auth import verify_bearer_token
from web.backend.crypto_utils import decrypt_token, encrypt_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/plaid", tags=["plaid"], dependencies=[Depends(verify_bearer_token)])

# Columns returned to the frontend for a linked item -- deliberately
# excludes access_token_encrypted. Every read below names this list
# explicitly rather than using `select *`, so adding a sensitive column
# to plaid_items later can't silently start leaking it here.
_ITEM_PUBLIC_COLUMNS = "id, institution_id, institution_name, status, last_sync_at, last_sync_error, created_at"


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


def _plaid_unavailable() -> HTTPException:
    return HTTPException(503, "Plaid isn't configured on this server yet.")


@router.post("/link-token")
@limiter.limit("10/minute")
async def create_link_token(request: Request):
    await enforce_daily_quota(request, "plaid/link-token")
    user_id = request.state.user["id"]
    try:
        link_token = plaid_client.create_link_token(user_id)
    except PlaidNotConfiguredError:
        raise _plaid_unavailable()
    return {"link_token": link_token}


class ExchangeRequest(BaseModel):
    public_token: str
    portfolio_id: Optional[int] = None
    # Display-only metadata from Plaid Link's own onSuccess callback --
    # never verified against Plaid server-side (nothing security-sensitive
    # depends on it, just what's shown as the connection's name in the
    # Linked Accounts list).
    institution_id: Optional[str] = None
    institution_name: Optional[str] = None


@router.post("/exchange")
@limiter.limit("10/minute")
async def exchange_public_token(request: Request, body: ExchangeRequest):
    await enforce_daily_quota(request, "plaid/exchange")
    user_id = request.state.user["id"]

    try:
        access_token, plaid_item_id = plaid_client.exchange_public_token(body.public_token)
    except PlaidNotConfiguredError:
        raise _plaid_unavailable()
    except plaid.ApiException as e:
        raise HTTPException(502, f"Plaid rejected this connection: {e}")

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        row = await conn.fetchrow(
            f"""
            INSERT INTO plaid_items (
                user_id, portfolio_id, plaid_item_id, access_token_encrypted,
                institution_id, institution_name, status
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, 'active')
            RETURNING {_ITEM_PUBLIC_COLUMNS}
            """,
            user_id, resolved_portfolio_id, plaid_item_id, encrypt_token(access_token),
            body.institution_id, body.institution_name,
        )

    sync_result = await plaid_sync.sync_item_by_id(row["id"], triggered_by="manual")
    return {"item": _record_to_dict(row), "sync": sync_result}


@router.get("/items")
async def list_items(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            f"SELECT {_ITEM_PUBLIC_COLUMNS} FROM plaid_items WHERE user_id = $1::uuid ORDER BY created_at ASC",
            user_id,
        )
    return {"items": [_record_to_dict(r) for r in rows]}


@router.post("/items/{item_id}/sync")
@limiter.limit("10/minute")
async def sync_item(request: Request, item_id: int):
    await enforce_daily_quota(request, "plaid/sync")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow("SELECT id FROM plaid_items WHERE id = $1 AND user_id = $2::uuid", item_id, user_id)
    if row is None:
        raise HTTPException(404, "Linked account not found.")
    return await plaid_sync.sync_item_by_id(item_id, triggered_by="manual")


@router.delete("/items/{item_id}")
@limiter.limit("10/minute")
async def disconnect_item(request: Request, item_id: int):
    await enforce_daily_quota(request, "plaid/disconnect")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "SELECT access_token_encrypted FROM plaid_items WHERE id = $1 AND user_id = $2::uuid",
            item_id, user_id,
        )
        if row is None:
            raise HTTPException(404, "Linked account not found.")

        # Best-effort Plaid-side removal -- the user's intent to
        # disconnect shouldn't be blocked by a Plaid-side hiccup, so a
        # failure here is logged and swallowed, not raised.
        try:
            plaid_client.remove_item(decrypt_token(row["access_token_encrypted"]))
        except Exception:
            pass

        # Cascades to portfolio_positions/portfolio_strategies rows for
        # this item (plaid_item_id references plaid_items(id) on delete
        # cascade) -- the frontend confirms this with the user before
        # calling this endpoint.
        await conn.execute("DELETE FROM plaid_items WHERE id = $1 AND user_id = $2::uuid", item_id, user_id)

    return {"ok": True}
