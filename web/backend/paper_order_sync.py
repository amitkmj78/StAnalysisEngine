"""
Scheduler-facing paper-trading reconciliation: order-status polling and
positions/cash sync. Modeled directly on web/backend/plaid_sync.py's
shape -- service_conn()-scoped (bypasses RLS, cross-user by design;
correctness comes from every write's own user_id/alpaca_paper_account_id
WHERE clause, never from RLS).

Deliberately writes BOTH portfolio_positions and portfolio_strategies, via
services/portfolio_strategy.build_robinhood_strategies -- same as
plaid_sync.sync_item_holdings. Skipping portfolio_strategies would repeat
a real bug already hit once in this app (a positions-only save left the
Holdings/Strategies UI empty, since that UI reads from portfolio_strategies).
Every write here is scoped by alpaca_paper_account_id, exactly like
plaid_sync's plaid_item_id scoping, so a sync only ever touches rows this
exact account previously wrote.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from starlette.concurrency import run_in_threadpool

from services import alpaca_trading_client
from services.alpaca_trading_client import AlpacaTradingError
from services.portfolio_strategy import build_robinhood_strategies
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn
from web.backend.routers.portfolio import _invalidate_insights_snapshot, _nan_to_none

logger = logging.getLogger(__name__)

_OPEN_STATUSES = ("SUBMITTING", "OPEN", "PARTIALLY_FILLED")

# Alpaca's own order.status values that mean "no longer working" -- mapped
# to this app's state machine. Anything not listed here (accepted,
# pending_new, ...) is treated as still OPEN.
_ALPACA_STATUS_MAP = {
    "filled": "FILLED",
    "partially_filled": "PARTIALLY_FILLED",
    "canceled": "CANCELLED",
    "expired": "CANCELLED",
    "rejected": "REJECTED",
}

_UNKNOWN_GRACE_SECONDS = 60


def reconcile_order_status(local: dict, broker_order: dict | None, now: datetime) -> dict | None:
    """Pure reconciliation decision: given a local paper_orders row and
    Alpaca's own view of that order (or None if Alpaca has no matching
    order), returns the fields to write if anything changed, or None if
    nothing changed / it's too soon to call it UNKNOWN. Kept side-effect
    free (no DB/HTTP) so it's unit-testable with synthetic dicts."""
    if broker_order is None:
        if local["alpaca_order_id"] is None:
            age_seconds = (now - local["created_at"]).total_seconds()
            if age_seconds > _UNKNOWN_GRACE_SECONDS:
                return {
                    "status": "UNKNOWN", "filled_qty": local["filled_qty"],
                    "filled_avg_price": local["filled_avg_price"], "reject_reason": None, "alpaca_order_id": None,
                }
        return None

    new_status = _ALPACA_STATUS_MAP.get(broker_order.get("status"), "OPEN")
    filled_qty = float(broker_order.get("filled_qty") or 0)
    filled_avg_price = float(broker_order["filled_avg_price"]) if broker_order.get("filled_avg_price") else None
    reject_reason = broker_order.get("rejected_reason") if new_status == "REJECTED" else None

    if (new_status, filled_qty, filled_avg_price) == (local["status"], local["filled_qty"], local["filled_avg_price"]):
        return None
    return {
        "status": new_status, "filled_qty": filled_qty, "filled_avg_price": filled_avg_price,
        "reject_reason": reject_reason, "alpaca_order_id": broker_order.get("id"),
    }


def _alpaca_positions_to_df(positions: list[dict]) -> pd.DataFrame:
    """Pure mapping from Alpaca's GET /v2/positions response to the
    Ticker/Shares/Avg_Cost contract build_robinhood_strategies() expects
    -- same contract services/plaid_holdings.py's holdings_to_positions()
    produces, so this plugs into the same downstream save shape."""
    rows = [
        {
            "Ticker": str(p["symbol"]).strip().upper(),
            "Shares": float(p["qty"]),
            "Avg_Cost": float(p["avg_entry_price"]),
        }
        for p in positions
        if float(p.get("qty", 0)) > 0
    ]
    if not rows:
        return pd.DataFrame(columns=["Ticker", "Shares", "Avg_Cost"])
    return pd.DataFrame(rows)


async def _mark_account_status(account_id: int, status: str, error_detail: str | None) -> None:
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE alpaca_paper_accounts SET status = $1, last_sync_error = $2 WHERE id = $3",
            status, error_detail, account_id,
        )


async def _audit(user_id: str, paper_order_id: int | None, event_type: str, detail: dict) -> None:
    import json
    async with service_conn() as conn:
        await conn.execute(
            "INSERT INTO paper_order_audit_log (user_id, paper_order_id, event_type, detail) VALUES ($1::uuid, $2, $3, $4::jsonb)",
            user_id, paper_order_id, event_type, json.dumps(detail, default=str),
        )


async def _fetch_accounts_with_open_orders() -> list[dict]:
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT a.*
            FROM alpaca_paper_accounts a
            JOIN paper_orders o ON o.alpaca_paper_account_id = a.id
            WHERE o.status = ANY($1::text[]) AND a.status = 'active'
            """,
            list(_OPEN_STATUSES),
        )
    return [dict(r) for r in rows]


async def poll_open_orders() -> int:
    """Scheduler entry point. Reconciles every account with at least one
    order still in an open-ish local status against Alpaca's own view.
    Returns the number of orders whose status changed."""
    accounts = await _fetch_accounts_with_open_orders()
    changed = 0
    for account in accounts:
        try:
            secret_key = decrypt_token(account["api_secret_key_encrypted"])
        except ValueError:
            await _mark_account_status(account["id"], "invalid_key", "Could not decrypt stored key.")
            continue

        async with service_conn() as conn:
            local_orders = await conn.fetch(
                "SELECT * FROM paper_orders WHERE alpaca_paper_account_id = $1 AND status = ANY($2::text[])",
                account["id"], list(_OPEN_STATUSES),
            )

        try:
            broker_orders = await run_in_threadpool(
                alpaca_trading_client.list_open_orders, account["api_key_id"], secret_key
            )
        except AlpacaTradingError as e:
            logger.warning("Alpaca list_open_orders failed for account %s: %s", account["id"], e.message)
            if e.status_code in (401, 403):
                await _mark_account_status(account["id"], "invalid_key", e.message)
            continue

        by_alpaca_id = {o["id"]: o for o in broker_orders}

        for local in local_orders:
            local = dict(local)
            broker_order = by_alpaca_id.get(local["alpaca_order_id"]) if local["alpaca_order_id"] else None

            if broker_order is None and local["alpaca_order_id"]:
                # Not in the open-orders list anymore -- either just
                # closed (filled/rejected/cancelled) or genuinely missing.
                try:
                    broker_order = await run_in_threadpool(
                        alpaca_trading_client.get_order, account["api_key_id"], secret_key, local["alpaca_order_id"]
                    )
                except AlpacaTradingError:
                    broker_order = None

            update = reconcile_order_status(local, broker_order, datetime.now(timezone.utc))
            if update is not None:
                await _apply_order_update(
                    local, update["status"], update["filled_qty"], update["filled_avg_price"],
                    update["reject_reason"], alpaca_order_id=update["alpaca_order_id"],
                )
                await _audit(local["user_id"], local["id"], "status_change", {"from": local["status"], "to": update["status"], **update})
                changed += 1

        await sync_positions_and_cash_for_account(account)

    return changed


async def _apply_order_update(local: dict, status: str, filled_qty: float, filled_avg_price, reject_reason, alpaca_order_id=None) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            UPDATE paper_orders
            SET status = $1, filled_qty = $2, filled_avg_price = $3, reject_reason = $4,
                alpaca_order_id = COALESCE($5, alpaca_order_id), last_polled_at = now()
            WHERE id = $6
            """,
            status, filled_qty, filled_avg_price, reject_reason, alpaca_order_id, local["id"],
        )


async def sync_positions_and_cash_for_account(account: dict) -> int:
    """Mirrors plaid_sync.sync_item_holdings's delete+reinsert-both-tables
    shape exactly, scoped by alpaca_paper_account_id instead of
    plaid_item_id, source='AlpacaPaper'."""
    try:
        secret_key = decrypt_token(account["api_secret_key_encrypted"])
    except ValueError:
        await _mark_account_status(account["id"], "invalid_key", "Could not decrypt stored key.")
        return 0

    try:
        positions = await run_in_threadpool(
            alpaca_trading_client.list_positions, account["api_key_id"], secret_key
        )
    except AlpacaTradingError as e:
        logger.warning("Alpaca list_positions failed for account %s: %s", account["id"], e.message)
        await _mark_account_status(account["id"], "invalid_key" if e.status_code in (401, 403) else "active", e.message)
        return 0

    positions_df = _alpaca_positions_to_df(positions)
    strat_df = (
        await run_in_threadpool(build_robinhood_strategies, positions_df) if not positions_df.empty else positions_df
    )

    user_id = account["user_id"]
    portfolio_id = account["portfolio_id"]
    account_id = account["id"]
    inserted = 0

    async with service_conn() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND alpaca_paper_account_id = $3",
                user_id, portfolio_id, account_id,
            )
            await conn.execute(
                "DELETE FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2 AND alpaca_paper_account_id = $3",
                user_id, portfolio_id, account_id,
            )
            for _, row in strat_df.iterrows():
                await conn.execute(
                    """
                    INSERT INTO portfolio_positions (
                        user_id, portfolio_id, ticker, name, shares, avg_cost, current_price,
                        unrealized_pnl_pct, source, alpaca_paper_account_id
                    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, 'AlpacaPaper', $9)
                    """,
                    user_id, portfolio_id, row["Ticker"], row["Ticker"], _nan_to_none(row["Shares"]),
                    _nan_to_none(row["Avg_Cost"]), _nan_to_none(row["Current_Price"]),
                    _nan_to_none(row["Unrealized_PnL_%"]), account_id,
                )
                await conn.execute(
                    """
                    INSERT INTO portfolio_strategies (
                        user_id, portfolio_id, ticker, shares, avg_cost, current_price, unrealized_pnl_pct,
                        short_term_plan, long_term_plan, risk_profile, risk_factor, alpaca_paper_account_id
                    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    user_id, portfolio_id, row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
                    _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]),
                    row["Short_Term_Plan"], row["Long_Term_Plan"], row["Risk_Profile"], int(row["Risk_Factor"]), account_id,
                )
                inserted += 1
            await conn.execute(
                "UPDATE alpaca_paper_accounts SET last_sync_at = now(), last_sync_error = null WHERE id = $1",
                account_id,
            )
            await _invalidate_insights_snapshot(conn, user_id, portfolio_id)

    return inserted


async def sync_positions_and_cash() -> int:
    """Scheduler entry point -- every active linked paper account."""
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM alpaca_paper_accounts WHERE status = 'active'")
    total = 0
    for row in rows:
        total += await sync_positions_and_cash_for_account(dict(row))
    return total
