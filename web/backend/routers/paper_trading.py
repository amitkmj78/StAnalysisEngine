"""
Paper trading (Alpaca), Stage 1: link a paper API key pair, accept the
paper-mode disclosure, submit market/limit orders through app-side
pre-trade checks with an idempotent submit, and list orders. Cancel/modify
and stop/stop-limit order types are Stage 2 (not yet built).

Every write here is real (paper) money moving through a real Alpaca
sandbox account -- there is no live-trading base URL anywhere in
services/alpaca_trading_client.py, so there is no way for this endpoint
to accidentally reach a live account.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services import alpaca_trading_client, paper_trading_checks
from services.alpaca_client import get_alpaca_latest_price
from services.alpaca_trading_client import AlpacaTradingError
from web.backend.app_settings import (
    PAPER_TRADING_ENABLED_KEY,
    PAPER_TRADING_KILL_SWITCH_KEY,
    PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT,
    PAPER_TRADING_MAX_ORDER_VALUE_KEY,
    PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT,
    PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY,
    PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT,
    PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY,
    PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT,
    PAPER_TRADING_PRICE_COLLAR_PCT_KEY,
    PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT,
    PAPER_TRADING_RESTRICTED_SYMBOLS_KEY,
    get_setting_bool,
    get_setting_float,
    get_setting_int,
    get_setting_str,
)
from web.backend.auth import verify_bearer_token
from web.backend.crypto_utils import decrypt_token, encrypt_token
from web.backend.db import user_conn
from web.backend.paper_order_sync import sync_positions_and_cash_for_account
from web.backend.rate_limit import enforce_daily_quota, limiter
from web.backend.routers.portfolio import _resolve_portfolio_id

router = APIRouter(prefix="/api/v1/paper-trading", tags=["paper-trading"], dependencies=[Depends(verify_bearer_token)])

# Columns returned to the frontend -- never api_secret_key_encrypted.
_ACCOUNT_PUBLIC_COLUMNS = (
    "id, portfolio_id, api_key_id, alpaca_account_id, account_number, status, "
    "last_sync_at, last_sync_error, disclosure_accepted_at, created_at"
)


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


async def _audit(user_id: str, paper_order_id: Optional[int], event_type: str, detail: dict) -> None:
    async with user_conn(user_id) as conn:
        await conn.execute(
            "INSERT INTO paper_order_audit_log (user_id, paper_order_id, event_type, detail) VALUES ($1::uuid, $2, $3, $4::jsonb)",
            user_id, paper_order_id, event_type, json.dumps(detail, default=str),
        )


async def _get_account_or_404(conn, user_id: str) -> dict:
    row = await conn.fetchrow(
        f"SELECT {_ACCOUNT_PUBLIC_COLUMNS}, api_secret_key_encrypted FROM alpaca_paper_accounts WHERE user_id = $1::uuid",
        user_id,
    )
    if row is None:
        raise HTTPException(404, "No paper-trading account linked yet.")
    return dict(row)


class LinkRequest(BaseModel):
    api_key_id: str = Field(min_length=1)
    api_secret_key: str = Field(min_length=1)
    portfolio_id: Optional[int] = None


@router.post("/link")
@limiter.limit("10/minute")
async def link_paper_account(request: Request, body: LinkRequest):
    await enforce_daily_quota(request, "paper-trading/link")
    user_id = request.state.user["id"]

    try:
        account_info = await run_in_threadpool(alpaca_trading_client.get_account, body.api_key_id, body.api_secret_key)
    except AlpacaTradingError as e:
        raise HTTPException(422, f"Alpaca rejected this key pair: {e.message}")

    secret_encrypted = encrypt_token(body.api_secret_key)
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        existing = await conn.fetchval("SELECT id FROM alpaca_paper_accounts WHERE user_id = $1::uuid", user_id)
        if existing is not None:
            raise HTTPException(409, "A paper-trading account is already linked. Unlink it first to link a different one.")
        row = await conn.fetchrow(
            f"""
            INSERT INTO alpaca_paper_accounts (
                user_id, portfolio_id, api_key_id, api_secret_key_encrypted, alpaca_account_id, account_number, status
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, 'active')
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            user_id, resolved_portfolio_id, body.api_key_id, secret_encrypted,
            account_info.get("id"), account_info.get("account_number"),
        )

    await _audit(user_id, None, "account_linked", {"alpaca_account_id": account_info.get("id")})
    sync_count = await sync_positions_and_cash_for_account({**_record_to_dict(row), "api_secret_key_encrypted": secret_encrypted})
    return {"account": _record_to_dict(row), "positions_synced": sync_count}


@router.post("/disclosure-accept")
async def accept_disclosure(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            f"""
            UPDATE alpaca_paper_accounts SET disclosure_accepted_at = now()
            WHERE user_id = $1::uuid
            RETURNING {_ACCOUNT_PUBLIC_COLUMNS}
            """,
            user_id,
        )
    if row is None:
        raise HTTPException(404, "No paper-trading account linked yet.")
    return {"account": _record_to_dict(row)}


@router.get("/account")
async def get_account(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        account = await _get_account_or_404(conn, user_id)

    secret_key = decrypt_token(account["api_secret_key_encrypted"])
    try:
        live = await run_in_threadpool(alpaca_trading_client.get_account, account["api_key_id"], secret_key)
    except AlpacaTradingError as e:
        live = None
        if e.status_code in (401, 403):
            async with user_conn(user_id) as conn:
                await conn.execute("UPDATE alpaca_paper_accounts SET status = 'invalid_key' WHERE user_id = $1::uuid", user_id)

    account.pop("api_secret_key_encrypted", None)
    return {"account": account, "live": live}


@router.get("/clock")
async def get_clock(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        account = await _get_account_or_404(conn, user_id)
    secret_key = decrypt_token(account["api_secret_key_encrypted"])
    try:
        return await run_in_threadpool(alpaca_trading_client.get_clock, account["api_key_id"], secret_key)
    except AlpacaTradingError as e:
        raise HTTPException(502, f"Could not reach Alpaca: {e.message}")


@router.delete("/link")
@limiter.limit("10/minute")
async def unlink_paper_account(request: Request):
    await enforce_daily_quota(request, "paper-trading/unlink")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchval("DELETE FROM alpaca_paper_accounts WHERE user_id = $1::uuid RETURNING id", user_id)
    if row is None:
        raise HTTPException(404, "No paper-trading account linked yet.")
    return {"ok": True}


class OrderRequest(BaseModel):
    ticker: str = Field(min_length=1)
    side: str = Field(pattern="^(buy|sell)$")
    order_type: str = Field(pattern="^(market|limit)$")
    time_in_force: str = Field(pattern="^(day|gtc)$")
    qty: float = Field(gt=0)
    limit_price: Optional[float] = Field(default=None, gt=0)


@router.get("/orders")
async def list_orders(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT * FROM paper_orders WHERE user_id = $1::uuid ORDER BY created_at DESC LIMIT 200", user_id
        )
    return {"orders": [_record_to_dict(r) for r in rows]}


@router.get("/orders/{order_id}")
async def get_order(request: Request, order_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        order = await conn.fetchrow("SELECT * FROM paper_orders WHERE id = $1 AND user_id = $2::uuid", order_id, user_id)
        if order is None:
            raise HTTPException(404, "Order not found.")
        audit_rows = await conn.fetch(
            "SELECT * FROM paper_order_audit_log WHERE paper_order_id = $1 ORDER BY created_at ASC", order_id
        )
    return {"order": _record_to_dict(order), "audit_log": [_record_to_dict(r) for r in audit_rows]}


@router.post("/orders")
@limiter.limit("10/minute")
async def submit_order(request: Request, body: OrderRequest):
    await enforce_daily_quota(request, "paper-trading/orders")
    user_id = request.state.user["id"]

    # Kill switch first -- before this touches the DB or Alpaca at all.
    if await get_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, default=False):
        await _audit(user_id, None, "kill_switch_blocked", {"ticker": body.ticker, "side": body.side})
        raise HTTPException(423, "Paper trading is temporarily paused by an administrator. Try again later.")

    if not await get_setting_bool(PAPER_TRADING_ENABLED_KEY, default=False):
        raise HTTPException(503, "Paper trading isn't enabled yet.")

    async with user_conn(user_id) as conn:
        account = await _get_account_or_404(conn, user_id)
        if account["disclosure_accepted_at"] is None:
            raise HTTPException(403, "You must accept the paper-trading disclosures before placing an order.")

        orders_today_count = await conn.fetchval(
            "SELECT count(*) FROM paper_orders WHERE user_id = $1::uuid AND created_at >= $2",
            user_id, datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        )
        existing_shares = await conn.fetchval(
            "SELECT shares FROM portfolio_positions WHERE user_id = $1::uuid AND alpaca_paper_account_id = $2 AND ticker = $3",
            user_id, account["id"], body.ticker.upper(),
        )

    secret_key = decrypt_token(account["api_secret_key_encrypted"])

    try:
        clock = await run_in_threadpool(alpaca_trading_client.get_clock, account["api_key_id"], secret_key)
    except AlpacaTradingError as e:
        raise HTTPException(503, f"Trading is unavailable right now: {e.message}")
    if not clock.get("is_open"):
        raise HTTPException(422, "The market is closed. Extended-hours orders aren't supported yet.")

    try:
        live_account = await run_in_threadpool(alpaca_trading_client.get_account, account["api_key_id"], secret_key)
    except AlpacaTradingError as e:
        raise HTTPException(503, f"Could not fetch your account balances: {e.message}")

    try:
        last_price = await run_in_threadpool(get_alpaca_latest_price, body.ticker.upper())
    except Exception:
        last_price = None

    max_order_value = await get_setting_float(PAPER_TRADING_MAX_ORDER_VALUE_KEY, default=PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT)
    max_pct = await get_setting_float(PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY, default=PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT)
    max_orders_per_day = await get_setting_int(PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY, default=PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT)
    collar_pct = await get_setting_float(PAPER_TRADING_PRICE_COLLAR_PCT_KEY, default=PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT)
    restricted_raw = await get_setting_str(PAPER_TRADING_RESTRICTED_SYMBOLS_KEY, default=PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT)
    restricted_symbols = {t.strip().upper() for t in restricted_raw.split(",") if t.strip()}

    estimate = paper_trading_checks.estimate_order_value(body.qty, body.order_type, body.limit_price, last_price)
    equity = float(live_account.get("equity", 0))

    checks = [
        paper_trading_checks.check_restricted_symbol(body.ticker, restricted_symbols),
        paper_trading_checks.check_buying_power(body.side, body.qty, estimate, live_account, float(existing_shares or 0)),
        paper_trading_checks.check_order_value_limit(estimate, max_order_value),
        paper_trading_checks.check_portfolio_pct_limit(estimate, equity, max_pct),
        paper_trading_checks.check_daily_order_count(int(orders_today_count), max_orders_per_day),
        paper_trading_checks.check_price_collar(body.order_type, body.limit_price, last_price, collar_pct),
    ]
    failures = [c for c in checks if not c.passed]
    if failures:
        await _audit(user_id, None, "check_failed", {"failures": [{"code": c.code, "message": c.message} for c in failures]})
        # A single joined string, matching every other endpoint's `detail`
        # contract (web/frontend/lib/api.ts reads body.detail as a plain
        # string) -- each check's own message is already plain-language
        # (TRD-25's spirit applied to pre-trade checks too).
        raise HTTPException(422, " ".join(c.message for c in failures))

    client_order_id = str(uuid.uuid4())
    async with user_conn(user_id) as conn:
        order_row = await conn.fetchrow(
            """
            INSERT INTO paper_orders (
                user_id, alpaca_paper_account_id, client_order_id, ticker, side, order_type,
                time_in_force, qty, limit_price, status, submitted_at
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, 'SUBMITTING', now())
            RETURNING *
            """,
            user_id, account["id"], client_order_id, body.ticker.upper(), body.side, body.order_type,
            body.time_in_force, body.qty, body.limit_price,
        )
    await _audit(user_id, order_row["id"], "submit_attempt", {"client_order_id": client_order_id})

    try:
        broker_order = await run_in_threadpool(
            alpaca_trading_client.submit_order,
            account["api_key_id"], secret_key,
            client_order_id=client_order_id, ticker=body.ticker.upper(), side=body.side,
            qty=body.qty, order_type=body.order_type, time_in_force=body.time_in_force,
            limit_price=body.limit_price,
        )
    except AlpacaTradingError as e:
        # Ambiguous vs. definite: a 4xx from Alpaca is a definite answer
        # (this specific submit was refused). Anything else that reached
        # us as an AlpacaTradingError is still a real HTTP response, so
        # it's definite too -- only a raw network/timeout exception
        # (uncaught here, see below) is genuinely ambiguous.
        async with user_conn(user_id) as conn:
            await conn.execute(
                "UPDATE paper_orders SET status = 'REJECTED', reject_reason = $1 WHERE id = $2",
                e.message, order_row["id"],
            )
        await _audit(user_id, order_row["id"], "broker_response", {"status": "REJECTED", "reason": e.message})
        raise HTTPException(422, f"Alpaca rejected this order: {e.message}")
    except Exception:
        # Ambiguous failure (timeout/connection error) -- resolve via
        # Alpaca's own idempotency key before ever telling the caller to
        # retry, per TRD-20/NFR-2. Never blind-retry-submit.
        try:
            reconciled = await run_in_threadpool(
                alpaca_trading_client.get_order_by_client_order_id, account["api_key_id"], secret_key, client_order_id
            )
        except AlpacaTradingError:
            reconciled = None
        if reconciled is not None:
            async with user_conn(user_id) as conn:
                await conn.execute(
                    "UPDATE paper_orders SET status = 'OPEN', alpaca_order_id = $1 WHERE id = $2",
                    reconciled.get("id"), order_row["id"],
                )
            await _audit(user_id, order_row["id"], "broker_response", {"status": "OPEN", "reconciled": True})
            return {"order": {**_record_to_dict(order_row), "status": "OPEN"}, "note": "Submitted (confirmed after a network delay)."}
        await _audit(user_id, order_row["id"], "broker_response", {"status": "SUBMITTING", "note": "no response yet, left for the poller"})
        return {"order": _record_to_dict(order_row), "note": "Your order is still being confirmed. Check back shortly."}

    async with user_conn(user_id) as conn:
        updated = await conn.fetchrow(
            "UPDATE paper_orders SET status = 'OPEN', alpaca_order_id = $1 WHERE id = $2 RETURNING *",
            broker_order.get("id"), order_row["id"],
        )
    await _audit(user_id, order_row["id"], "broker_response", {"status": "OPEN", "alpaca_order_id": broker_order.get("id")})
    return {"order": _record_to_dict(updated)}
