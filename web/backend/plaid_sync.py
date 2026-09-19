"""
Reconciles one Plaid-linked item's holdings into portfolio_positions /
portfolio_strategies. Modeled on web/backend/pit_prices.py's
capture_and_persist_* shape and web/backend/portfolio_alerts.py's
scan_portfolios_for_drops precedent for a cross-user background job that
writes into RLS-protected, user-owned tables via service_conn (RLS
bypassed; correctness here comes entirely from every write's own
user_id/portfolio_id/plaid_item_id WHERE clause, never from RLS).

Deliberately does NOT reuse web/backend/routers/portfolio.py's
_merge_with_existing/_save_and_respond: that path is ticker-keyed across
an entire portfolio, so it can neither detect a holding sold off at the
brokerage (a removed ticker just silently stays forever) nor guard
against a Plaid row and a manual row for the same ticker overwriting each
other. Every write below is scoped by plaid_item_id instead, so a sync
only ever touches rows this exact item previously wrote.
"""

from __future__ import annotations

import logging

from starlette.concurrency import run_in_threadpool

from services import plaid_client
from services.plaid_client import PlaidItemLoginRequiredError
from services.plaid_holdings import holdings_to_positions
from services.portfolio_strategy import build_robinhood_strategies
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn
from web.backend.routers.portfolio import _invalidate_insights_snapshot, _nan_to_none
from services.acquired_at_utils import acquired_at_or_today

logger = logging.getLogger(__name__)


async def _mark_item_status(item_id: int, status: str, error_detail: str | None) -> None:
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE plaid_items SET status = $1, last_sync_error = $2 WHERE id = $3",
            status, error_detail, item_id,
        )


async def _log_sync(user_id: str, item_id: int, status: str, positions_upserted: int | None, error_detail: str | None, triggered_by: str) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO plaid_sync_log (user_id, plaid_item_id, finished_at, status, positions_upserted, error_detail, triggered_by)
            VALUES ($1::uuid, $2, now(), $3, $4, $5, $6)
            """,
            user_id, item_id, status, positions_upserted, error_detail, triggered_by,
        )


async def sync_item_holdings(item: dict, triggered_by: str) -> dict:
    """
    item: a plaid_items row dict with a decrypted 'access_token' key
    already attached (see sync_item_by_id/sync_all_due_items below --
    kept separate from this function so tests can exercise the
    reconciliation logic without ever handling a real/fake token).

    On ITEM_LOGIN_REQUIRED or any other Plaid-side failure: does NOT
    touch portfolio_positions/portfolio_strategies at all -- stale-but-
    present beats silently wiping a portfolio to empty on an auth
    hiccup or a transient API error. Only plaid_items.status/
    last_sync_error changes, and the attempt is logged either way.
    """
    user_id = item["user_id"]
    portfolio_id = item["portfolio_id"]
    item_id = item["id"]

    try:
        holdings_response = await run_in_threadpool(plaid_client.fetch_holdings, item["access_token"])
    except PlaidItemLoginRequiredError as e:
        await _mark_item_status(item_id, "login_required", str(e))
        await _log_sync(user_id, item_id, "error", None, "ITEM_LOGIN_REQUIRED", triggered_by)
        return {"status": "login_required", "positions_upserted": 0}
    except Exception as e:  # Plaid ApiException or any transport failure
        logger.warning("Plaid sync failed for item %s: %s", item_id, e)
        await _mark_item_status(item_id, "error", str(e))
        await _log_sync(user_id, item_id, "error", None, str(e), triggered_by)
        return {"status": "error", "positions_upserted": 0}

    positions_df = holdings_to_positions(holdings_response)
    strat_df = (
        await run_in_threadpool(build_robinhood_strategies, positions_df) if not positions_df.empty else positions_df
    )

    inserted = 0
    async with service_conn() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND plaid_item_id = $3",
                user_id, portfolio_id, item_id,
            )
            await conn.execute(
                "DELETE FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2 AND plaid_item_id = $3",
                user_id, portfolio_id, item_id,
            )
            for _, row in strat_df.iterrows():
                await conn.execute(
                    """
                    INSERT INTO portfolio_positions (
                        user_id, portfolio_id, ticker, name, shares, avg_cost, current_price,
                        unrealized_pnl_pct, source, acquired_at, plaid_item_id
                    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, 'Plaid', $9, $10)
                    """,
                    user_id, portfolio_id, row["Ticker"], row["Ticker"], _nan_to_none(row["Shares"]),
                    _nan_to_none(row["Avg_Cost"]), _nan_to_none(row["Current_Price"]),
                    _nan_to_none(row["Unrealized_PnL_%"]), acquired_at_or_today(row.get("Acquired_At")), item_id,
                )
                await conn.execute(
                    """
                    INSERT INTO portfolio_strategies (
                        user_id, portfolio_id, ticker, shares, avg_cost, current_price, unrealized_pnl_pct,
                        short_term_plan, long_term_plan, risk_profile, risk_factor, plaid_item_id
                    ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    user_id, portfolio_id, row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
                    _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]),
                    row["Short_Term_Plan"], row["Long_Term_Plan"], row["Risk_Profile"], int(row["Risk_Factor"]), item_id,
                )
                inserted += 1
            await conn.execute(
                "UPDATE plaid_items SET last_sync_at = now(), status = 'active', last_sync_error = null WHERE id = $1",
                item_id,
            )
            await _invalidate_insights_snapshot(conn, user_id, portfolio_id)

    await _log_sync(user_id, item_id, "success", inserted, None, triggered_by)
    return {"status": "success", "positions_upserted": inserted}


async def _fetch_items(where_clause: str, params: list) -> list[dict]:
    async with service_conn() as conn:
        rows = await conn.fetch(f"SELECT * FROM plaid_items WHERE {where_clause}", *params)
    return [dict(r) for r in rows]


async def _sync_items(items: list[dict], triggered_by: str) -> list[dict]:
    results = []
    for item in items:
        item["access_token"] = decrypt_token(item["access_token_encrypted"])
        results.append(await sync_item_holdings(item, triggered_by=triggered_by))
    return results


async def sync_item_by_id(item_id: int, triggered_by: str) -> dict:
    """Router-facing single-item sync ('Sync Now' / post-Link initial
    sync) -- the router only ever verifies ownership via user_conn and
    hands over a bare item_id, never a token or the encrypted column."""
    items = await _fetch_items("id = $1 AND status != 'revoked'", [item_id])
    if not items:
        return {"status": "not_found", "positions_upserted": 0}
    return (await _sync_items(items, triggered_by))[0]


async def sync_all_due_items() -> int:
    """Scheduler entry point -- every active linked item."""
    items = await _fetch_items("status != 'revoked'", [])
    results = await _sync_items(items, triggered_by="scheduler")
    return sum(r["positions_upserted"] for r in results)
