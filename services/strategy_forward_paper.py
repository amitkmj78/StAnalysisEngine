"""STB-6: a saved strategy run forward in paper mode, with its own real
(never live) brokered paper account and live track record.

Order of operations per strategy_paper_accounts row, same fail-safe shape
as services/agent/runner.py::run_agent_for_user:
  1. Not enabled / disclosure not accepted -> nothing happens.
  2. Kill switch (row-level, global STRATEGY_FORWARD_PAPER_KILL_SWITCH_KEY,
     or the shared PAPER_TRADING_KILL_SWITCH_KEY) -> journal 'skipped'.
  3. Load the owning saved_strategies.definition, parse via the exact same
     Rule.parse/ProtectiveExits.parse POST /backtest uses.
  4. Resolve tickers via strategy_forward_record's own ticker-resolution
     logic (reused, not reimplemented).
  5. Per ticker: build a feature frame from cached daily history, decide
     today_signal() against the open strategy_paper_positions row (or
     none), journal every proposed action BEFORE anything is sent.
  6. Broker clock closed -> plan journaled, nothing submitted.
  7. Submit sells before buys. Each order runs through every
     services/paper_trading_checks.py function -- the SAME bar a human
     submitting a manual ticket clears (services/agent/runner.py's own
     order path does not do this; this job deliberately holds itself to a
     higher bar). A failing check journals the rejection and is skipped,
     never retried.
  8. Submit via alpaca_trading_client.submit_order with a fresh
     client_order_id; an ambiguous network failure is reconciled via
     get_order_by_client_order_id before ever retrying.
  9. A submitted sell clears the position and starts its cooldown; a
     submitted buy records entry_price/entry_atr/peak at the price used
     to size it (NOT polled for an actual fill price -- a disclosed v1
     simplification, same spirit as other estimate-based sizing already
     in this codebase).

Known, disclosed residual risk: unlike services/agent/runner.py's
_ensure_stops, this job does NOT guarantee a broker-side stop order exists
between runs -- a saved strategy's exits (which can combine atr_stop_k +
stop_loss_pct + take_profit_pct + time_stop_sessions) have no single
Alpaca native order type to rest on. Exits are enforced only when this job
runs, once a day -- stated plainly in the link/enable UI, not hidden.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
from starlette.concurrency import run_in_threadpool

from services import alpaca_trading_client, paper_trading_checks
from services.alpaca_client import get_alpaca_latest_price
from services.alpaca_trading_client import AlpacaTradingError
from services.challenge_service import compute_member_performance
from services.strategy_engine import DEFAULT_COOLDOWN, ProtectiveExits, Rule, feature_frame, today_signal
from services.strategy_forward_record import _resolve_tickers as resolve_tickers
from services.yfinance_cache import get_cached_history
from web.backend.app_settings import (
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
    STRATEGY_FORWARD_PAPER_KILL_SWITCH_KEY,
    get_setting_bool,
    get_setting_float,
    get_setting_int,
    get_setting_str,
)
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


def _f(value, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json(obj) -> str:
    return json.dumps(obj, default=str)


def _creds(account: dict) -> dict:
    return {"key": account["api_key_id"], "secret": decrypt_token(account["api_secret_key_encrypted"])}


async def _broker(fn, creds, *args, **kwargs):
    return await run_in_threadpool(fn, creds["key"], creds["secret"], *args, **kwargs)


# --- journal helpers (insert-only, mirror services/agent/runner.py exactly) ---

async def _write_run_header(account: dict, status: str, reason: Optional[str], equity: Optional[float] = None) -> int:
    async with service_conn() as conn:
        return await conn.fetchval(
            """
            INSERT INTO strategy_paper_runs (strategy_paper_account_id, saved_strategy_id, user_id, status, equity, reason)
            VALUES ($1, $2, $3::uuid, $4, $5, $6) RETURNING id
            """,
            account["id"], account["saved_strategy_id"], account["user_id"], status, equity, reason,
        )


async def _journal(run_id: int, account: dict, event_type: str, reason: str, **fields) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO strategy_paper_order_events
                (run_id, strategy_paper_account_id, user_id, event_type, ticker, side, qty, est_price, est_value,
                 reason, alpaca_order_id, client_order_id, detail)
            VALUES ($1, $2, $3::uuid, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
            """,
            run_id, account["id"], account["user_id"], event_type, fields.get("ticker"), fields.get("side"),
            fields.get("qty"), fields.get("est_price"), fields.get("est_value"), reason,
            fields.get("alpaca_order_id"), fields.get("client_order_id"), _json(fields.get("detail") or {}),
        )


async def _update_account_status(account_id: int, last_run_status: str) -> None:
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE strategy_paper_accounts SET last_run_at = now(), last_run_status = $2 WHERE id = $1",
            account_id, last_run_status,
        )


# --- state ---

async def _load_account(saved_strategy_id) -> Optional[dict]:
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM strategy_paper_accounts WHERE saved_strategy_id = $1", saved_strategy_id)
        return dict(row) if row else None


async def _load_definition(saved_strategy_id) -> Optional[dict]:
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT definition FROM saved_strategies WHERE id = $1", saved_strategy_id)
    if row is None:
        return None
    d = row["definition"]
    return json.loads(d) if isinstance(d, str) else d


async def _load_positions(account_id: int) -> dict[str, dict]:
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT * FROM strategy_paper_positions WHERE strategy_paper_account_id = $1", account_id)
    return {r["ticker"]: dict(r) for r in rows}


async def _upsert_position(account_id: int, ticker: str, *, qty: float, entry_price: Optional[float],
                           entry_atr: Optional[float], peak_price: Optional[float], entry_date: Optional[date],
                           days_held: int, cooldown_until_date: Optional[date]) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO strategy_paper_positions
                (strategy_paper_account_id, ticker, qty, entry_price, entry_atr, peak_price, entry_date, days_held, cooldown_until_date)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (strategy_paper_account_id, ticker) DO UPDATE SET
                qty = $3, entry_price = $4, entry_atr = $5, peak_price = $6, entry_date = $7,
                days_held = $8, cooldown_until_date = $9, updated_at = now()
            """,
            account_id, ticker, qty, entry_price, entry_atr, peak_price, entry_date, days_held, cooldown_until_date,
        )


async def _count_orders_today(account_id: int) -> int:
    async with service_conn() as conn:
        return await conn.fetchval(
            """
            SELECT count(*) FROM strategy_paper_order_events
            WHERE strategy_paper_account_id = $1 AND event_type = 'submitted' AND created_at >= $2
            """,
            account_id, datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc),
        )


# --- the run ---

async def run_forward_paper_for_strategy(saved_strategy_id, *, trigger: str = "scheduled") -> dict:
    account = await _load_account(saved_strategy_id)
    if account is None or not account["enabled"] or account["disclosure_accepted_at"] is None:
        return {"status": "not_enabled"}

    kill_engaged = account.get("kill_engaged") or await get_setting_bool(
        STRATEGY_FORWARD_PAPER_KILL_SWITCH_KEY, default=False
    ) or await get_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, default=False)
    if kill_engaged:
        run_id = await _write_run_header(account, "skipped", "Kill switch is engaged; no orders placed.")
        await _journal(run_id, account, "run_skipped", "Kill switch is engaged. No orders were placed.")
        await _update_account_status(account["id"], "skipped")
        return {"status": "skipped", "reason": "kill_switch", "run_id": run_id}

    try:
        creds = _creds(account)
    except ValueError:
        await _update_account_status(account["id"], "failed")
        return {"status": "failed", "error": "stored key unreadable"}

    try:
        return await _run_with_creds(account, creds, trigger)
    except Exception as e:  # noqa: BLE001 -- every failure must be journaled, never swallowed
        logger.exception("Strategy forward-paper run failed for strategy %s", saved_strategy_id)
        run_id = await _write_run_header(account, "failed", f"Run failed: {e}")
        await _journal(run_id, account, "run_failed", f"Run failed and was stopped: {e}")
        await _update_account_status(account["id"], "failed")
        return {"status": "failed", "error": str(e), "run_id": run_id}


async def _run_with_creds(account: dict, creds: dict, trigger: str) -> dict:
    definition = await _load_definition(account["saved_strategy_id"])
    if not definition:
        run_id = await _write_run_header(account, "skipped", "Saved strategy no longer exists or has no definition.")
        await _journal(run_id, account, "run_skipped", "Saved strategy no longer exists.")
        await _update_account_status(account["id"], "skipped")
        return {"status": "skipped", "reason": "no_definition", "run_id": run_id}

    entry_raw = definition.get("entry") or []
    exit_raw = definition.get("exit") or []
    if not entry_raw:
        run_id = await _write_run_header(account, "skipped", "Strategy has no entry rule.")
        await _journal(run_id, account, "run_skipped", "Strategy has no entry rule; nothing to evaluate.")
        await _update_account_status(account["id"], "skipped")
        return {"status": "skipped", "reason": "no_entry_rule", "run_id": run_id}

    entry = [Rule.parse(r) for r in entry_raw]
    exit_ = [Rule.parse(r) for r in exit_raw]
    exits = ProtectiveExits.parse(definition.get("exits"))
    cooldown = int(definition.get("cooldown_sessions") or 0) or DEFAULT_COOLDOWN

    universe: list[str] = []
    if not definition.get("weights") and definition.get("source") == "random_sample":
        from services.stock_finder_service import fetch_sp500_tickers
        try:
            universe = sorted(await run_in_threadpool(fetch_sp500_tickers))
        except Exception as e:  # noqa: BLE001 -- a failed universe fetch shouldn't crash the run
            logger.warning("Could not fetch S&P 500 membership for strategy %s: %s", account["saved_strategy_id"], e)
    tickers = resolve_tickers(definition, universe)
    if not tickers:
        run_id = await _write_run_header(account, "skipped", "No tickers resolved for this strategy's universe.")
        await _journal(run_id, account, "run_skipped", "No tickers resolved for this strategy's universe.")
        await _update_account_status(account["id"], "skipped")
        return {"status": "skipped", "reason": "no_tickers", "run_id": run_id}

    raw_weights = definition.get("weights")
    if raw_weights:
        total_w = sum(float(v) for v in raw_weights.values()) or 1.0
        weights = {k.strip().upper(): float(v) / total_w for k, v in raw_weights.items()}
    else:
        weights = {t: 1.0 / len(tickers) for t in tickers}

    clock = await _broker(alpaca_trading_client.get_clock, creds)
    market_open = bool(clock.get("is_open"))

    live_account = await _broker(alpaca_trading_client.get_account, creds)
    equity = _f(live_account.get("equity"))
    run_id = await _write_run_header(account, "started", f"Trigger: {trigger}.", equity)

    positions = await _load_positions(account["id"])
    today = date.today()

    decisions: dict[str, dict] = {}
    for ticker in tickers:
        held_row = positions.get(ticker)
        held = bool(held_row and _f(held_row.get("qty")) > 0)
        cooldown_active = bool(
            held_row and held_row.get("cooldown_until_date") and held_row["cooldown_until_date"] > today
        )
        try:
            history = await run_in_threadpool(get_cached_history, ticker, "2y", True)
        except Exception as e:  # noqa: BLE001 -- one bad ticker shouldn't block the rest
            await _journal(run_id, account, "skipped", f"Could not fetch price history for {ticker}: {e}", ticker=ticker)
            continue
        if history is None or history.empty:
            await _journal(run_id, account, "skipped", f"No price history for {ticker}.", ticker=ticker)
            continue

        frame = feature_frame(history)
        decision = today_signal(
            frame, entry, exit_, exits, held=held,
            entry_price=held_row.get("entry_price") if held_row else None,
            entry_atr=held_row.get("entry_atr") if held_row else None,
            peak=held_row.get("peak_price") if held_row else None,
            days_held=int(held_row.get("days_held") or 0) if held_row else 0,
            cooldown_active=cooldown_active,
        )
        last_close = float(frame["close"].iloc[-1])
        last_high = float(frame["high"].iloc[-1])
        last_atr = float(frame["atr_14"].iloc[-1]) if "atr_14" in frame.columns else None
        last_atr = last_atr if last_atr is not None and last_atr == last_atr else None  # NaN -> None

        if held and decision["action"] == "hold":
            # Carry position state forward one more day -- mirrors
            # _run_ticker's `peak = max(peak, highs[i])` and day-count.
            new_peak = max(_f(held_row.get("peak_price"), default=last_high) or last_high, last_high)
            await _upsert_position(
                account["id"], ticker, qty=_f(held_row.get("qty")), entry_price=held_row.get("entry_price"),
                entry_atr=held_row.get("entry_atr"), peak_price=new_peak, entry_date=held_row.get("entry_date"),
                days_held=int(held_row.get("days_held") or 0) + 1, cooldown_until_date=held_row.get("cooldown_until_date"),
            )

        decisions[ticker] = {
            "decision": decision, "last_close": last_close, "last_atr": last_atr, "held_row": held_row,
        }
        if decision["action"] != "hold":
            await _journal(
                run_id, account, "proposed", decision["reason"] or decision["action"],
                ticker=ticker, side="sell" if decision["action"] == "exit" else "buy",
            )

    if not market_open:
        await _journal(run_id, account, "run_completed", "Market is closed (broker clock): plan journaled, no orders placed.")
        await _update_account_status(account["id"], "completed")
        return {"status": "completed", "run_id": run_id, "orders_proposed": len(decisions), "orders_placed": 0}

    submitted = await _submit_decisions(run_id, account, creds, decisions, weights, equity, cooldown)
    await _journal(run_id, account, "run_completed", f"Run finished: {len(submitted)} order(s) submitted.")
    await _update_account_status(account["id"], "completed")
    return {"status": "completed", "run_id": run_id, "orders_placed": len(submitted)}


async def _submit_decisions(run_id: int, account: dict, creds: dict, decisions: dict, weights: dict,
                            equity: float, cooldown: int) -> list[dict]:
    max_order_value = await get_setting_float(PAPER_TRADING_MAX_ORDER_VALUE_KEY, default=PAPER_TRADING_MAX_ORDER_VALUE_DEFAULT)
    max_pct = await get_setting_float(PAPER_TRADING_MAX_PORTFOLIO_PCT_KEY, default=PAPER_TRADING_MAX_PORTFOLIO_PCT_DEFAULT)
    max_orders_per_day = await get_setting_int(PAPER_TRADING_MAX_ORDERS_PER_DAY_KEY, default=PAPER_TRADING_MAX_ORDERS_PER_DAY_DEFAULT)
    collar_pct = await get_setting_float(PAPER_TRADING_PRICE_COLLAR_PCT_KEY, default=PAPER_TRADING_PRICE_COLLAR_PCT_DEFAULT)
    restricted_raw = await get_setting_str(PAPER_TRADING_RESTRICTED_SYMBOLS_KEY, default=PAPER_TRADING_RESTRICTED_SYMBOLS_DEFAULT)
    restricted_symbols = {t.strip().upper() for t in restricted_raw.split(",") if t.strip()}
    orders_today_count = await _count_orders_today(account["id"])

    sells = {t: d for t, d in decisions.items() if d["decision"]["action"] == "exit"}
    buys = {t: d for t, d in decisions.items() if d["decision"]["action"] == "enter"}
    submitted: list[dict] = []

    for ticker, info in list(sells.items()) + list(buys.items()):
        side = "sell" if ticker in sells else "buy"
        order = await _submit_one(
            run_id, account, creds, ticker, side, info, weights, equity, cooldown,
            max_order_value, max_pct, max_orders_per_day, collar_pct, restricted_symbols, orders_today_count,
        )
        if order:
            orders_today_count += 1
            submitted.append(order)
    return submitted


async def _submit_one(run_id, account, creds, ticker: str, side: str, info: dict, weights: dict, equity: float,
                      cooldown: int, max_order_value: float, max_pct: float, max_orders_per_day: int,
                      collar_pct: float, restricted_symbols: set, orders_today_count: int) -> Optional[dict]:
    held_row = info["held_row"]
    last_close = info["last_close"]
    last_atr = info["last_atr"]

    try:
        last_price = await run_in_threadpool(get_alpaca_latest_price, ticker)
    except Exception:  # noqa: BLE001 -- fall back to the cached close rather than block the run
        last_price = None
    price = last_price or last_close

    if side == "sell":
        qty = _f(held_row.get("qty")) if held_row else 0.0
        shares_held = qty
    else:
        target_value = equity * weights.get(ticker, 0.0)
        qty = float(math.floor(target_value / price)) if price else 0.0
        shares_held = 0.0

    if not qty or qty <= 0:
        await _journal(run_id, account, "skipped", f"Computed quantity for {ticker} was zero; nothing to submit.", ticker=ticker, side=side)
        return None

    estimate = paper_trading_checks.estimate_order_value(qty, "market", None, price)
    checks = [
        paper_trading_checks.check_restricted_symbol(ticker, restricted_symbols),
        paper_trading_checks.check_buying_power(side, qty, estimate, {"buying_power": equity}, shares_held),
        paper_trading_checks.check_order_value_limit(estimate, max_order_value),
        paper_trading_checks.check_portfolio_pct_limit(estimate, equity, max_pct),
        paper_trading_checks.check_daily_order_count(orders_today_count, max_orders_per_day),
        paper_trading_checks.check_price_collar("market", None, price, collar_pct),
    ]
    failures = [c for c in checks if not c.passed]
    if failures:
        await _journal(run_id, account, "rejected", " ".join(c.message for c in failures), ticker=ticker, side=side, qty=qty)
        return None

    cid = str(uuid.uuid4())
    reason = info["decision"]["reason"] or info["decision"]["action"]
    await _journal(run_id, account, "submitting", reason, ticker=ticker, side=side, qty=qty,
                   est_price=price, est_value=estimate, client_order_id=cid)
    try:
        placed = await _broker(alpaca_trading_client.submit_order, creds, client_order_id=cid, ticker=ticker,
                               side=side, qty=qty, order_type="market", time_in_force="day")
    except AlpacaTradingError as e:
        await _journal(run_id, account, "rejected", f"Alpaca rejected the order: {e.message}",
                       ticker=ticker, side=side, qty=qty, client_order_id=cid)
        return None
    except httpx.HTTPError as e:
        recovered = await _broker(alpaca_trading_client.get_order_by_client_order_id, creds, cid)
        if recovered is None:
            await _journal(run_id, account, "rejected", f"Network failure before Alpaca accepted the order: {e}",
                           ticker=ticker, side=side, qty=qty, client_order_id=cid)
            return None
        placed = recovered

    await _journal(run_id, account, "submitted", reason, ticker=ticker, side=side, qty=qty,
                   alpaca_order_id=placed.get("id"), client_order_id=cid)

    today = date.today()
    if side == "sell":
        await _upsert_position(
            account["id"], ticker, qty=0.0, entry_price=None, entry_atr=None, peak_price=None,
            entry_date=None, days_held=0, cooldown_until_date=today + timedelta(days=cooldown),
        )
    else:
        await _upsert_position(
            account["id"], ticker, qty=qty, entry_price=price, entry_atr=last_atr, peak_price=price,
            entry_date=today, days_held=0, cooldown_until_date=None,
        )

    return {"ticker": ticker, "side": side, "alpaca_order_id": placed.get("id")}


# --- equity capture + track record ---

async def capture_equity_for_strategy_account(account: dict) -> bool:
    """Mirrors services/challenge_service.py::capture_equity_for_account's
    exact decrypt -> get_account -> insert shape, targeting
    strategy_paper_equity_snapshots instead of paper_account_equity_
    snapshots."""
    try:
        creds = _creds(account)
    except ValueError:
        return False
    try:
        live = await _broker(alpaca_trading_client.get_account, creds)
    except AlpacaTradingError as e:
        logger.warning("Strategy paper equity capture failed for account %s: %s", account["id"], e.message)
        return False
    equity = live.get("equity")
    if equity is None:
        return False
    async with service_conn() as conn:
        result = await conn.execute(
            """
            INSERT INTO strategy_paper_equity_snapshots (strategy_paper_account_id, saved_strategy_id, user_id, as_of_date, equity)
            VALUES ($1, $2, $3::uuid, $4, $5)
            ON CONFLICT (strategy_paper_account_id, as_of_date) DO NOTHING
            """,
            account["id"], account["saved_strategy_id"], account["user_id"], date.today(), float(equity),
        )
    return result == "INSERT 0 1"


async def evaluate_due_forward_paper_accounts() -> int:
    """Scheduler entry point. One attempt per enabled, disclosure-accepted
    strategy paper account; a failure on one strategy never aborts the
    rest (same per-item isolation as _run_trading_agents_job/
    evaluate_due_forward_records)."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT saved_strategy_id FROM strategy_paper_accounts WHERE enabled AND disclosure_accepted_at IS NOT NULL"
        )
    evaluated = 0
    for row in rows:
        try:
            await run_forward_paper_for_strategy(row["saved_strategy_id"], trigger="scheduled")
            evaluated += 1
        except Exception as e:  # noqa: BLE001 -- one bad strategy must never abort the batch
            logger.warning("Strategy forward-paper evaluation failed for strategy %s: %s", row["saved_strategy_id"], e)
    return evaluated


async def get_track_record(saved_strategy_id, user_id: str) -> dict:
    """Reuses services/challenge_service.py::compute_member_performance
    UNMODIFIED -- the same Sharpe/Sortino/Calmar/drawdown math every
    challenge leaderboard already computes on a real linked account, not a
    second implementation."""
    async with service_conn() as conn:
        account = await conn.fetchrow(
            "SELECT id, created_at FROM strategy_paper_accounts WHERE saved_strategy_id = $1 AND user_id = $2::uuid",
            saved_strategy_id, user_id,
        )
        if account is None:
            return {"available": False, "reason": "No paper account linked for this strategy."}
        rows = await conn.fetch(
            "SELECT as_of_date, equity FROM strategy_paper_equity_snapshots WHERE strategy_paper_account_id = $1",
            account["id"],
        )
    snapshots = [{"as_of_date": r["as_of_date"], "equity": r["equity"]} for r in rows]
    performance = compute_member_performance(snapshots, account["created_at"].date(), date.today())
    return {
        "available": True,
        "linked_at": account["created_at"].isoformat(),
        "equity_series": [
            {"date": s["as_of_date"].isoformat(), "equity": s["equity"]}
            for s in sorted(snapshots, key=lambda s: s["as_of_date"])
        ],
        **performance,
    }
