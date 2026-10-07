"""Trading agent run (Phase 4). One run per user per trading day.

Order of operations (fail-safe throughout):
  1. Load the user's agent settings. Not enabled -> nothing happens.
  2. Mode live -> every proposed order is journaled as rejected (AGT-2).
  3. Kill switch engaged -> journal skipped, place nothing (AGT-3).
  4. Broker clock: closed market -> skipped for orders (AGT-4). Plan runs
     may still compute a plan, since they place nothing.
  5. Read account, positions, regime, scores, price history.
  6. Risk state, candidate filters, sizing, plan (pure, services/agent/risk.py).
  7. Journal the header and every planned order before anything is sent.
  8. Paper mode only: re-check the kill switch, then submit sells before
     buys, each order through preflight(). Each attempt is journaled before
     it is sent, so an ambiguous network failure can be reconciled by
     client_order_id instead of blindly retried.
  9. Stop integrity (AGT-16/29): every open position must have a broker-side
     trailing stop sized to its current quantity. This also runs after a
     failed run, so a crash never leaves a position unprotected.
"""

import asyncio
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
from starlette.concurrency import run_in_threadpool

from services import alpaca_trading_client
from services.agent.config import CONFIG, config_version
from services.agent.indicators import annualized_volatility_pct, atr, average_dollar_volume, sma
from services.agent.risk import (
    Candidate,
    Holding,
    Order,
    filter_candidates,
    plan_orders,
    preflight,
    regime_cap_pct,
    risk_state,
    scale_for_vol_target,
    size_positions,
)
from services.alpaca_trading_client import AlpacaTradingError
from services.market_regime_service import regime_as_of
from services.notification_dispatcher import EASTERN, dispatch_alert
from services.stock_detail_service import upcoming_earnings_in_window
from services.yfinance_cache import get_cached_earnings_dates, get_cached_history
from web.backend.app_settings import AGENT_KILL_SWITCH_KEY, PAPER_TRADING_KILL_SWITCH_KEY, get_setting_bool
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

UNIVERSE_ID = "All"
FILL_WAIT_SECONDS = 60
FILL_POLL_SECONDS = 5
TERMINAL_EVENTS = ("filled", "canceled", "rejected", "expired")
LIVE_BLOCK_REASON = (
    "Live trading is blocked: the validation gate (AGT-30) has not passed and ALLOW_LIVE_TRADING is not set. "
    "No live order was placed."
)


# --- journal helpers (insert-only) ---

async def _write_run_header(user_id: str, mode: str, status: str, reason: Optional[str], snap: dict) -> int:
    async with service_conn() as conn:
        return await conn.fetchval(
            """
            INSERT INTO agent_runs (user_id, mode, status, regime, exposure_cap_pct, equity, last_equity,
                                    risk_state, config_version, reason)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id
            """,
            user_id, mode, status, snap.get("regime"), snap.get("exposure_cap_pct"), snap.get("equity"),
            snap.get("last_equity"), snap.get("risk_state"), config_version(), reason,
        )


async def _journal(run_id: int, user_id: str, event_type: str, reason: str, **fields) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO agent_order_events (run_id, user_id, event_type, ticker, side, qty, est_price, est_value,
                                            trigger, reason, alpaca_order_id, client_order_id, detail)
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
            """,
            run_id, user_id, event_type, fields.get("ticker"), fields.get("side"), fields.get("qty"),
            fields.get("est_price"), fields.get("est_value"), fields.get("trigger"), reason,
            fields.get("alpaca_order_id"), fields.get("client_order_id"),
            _json(fields.get("detail") or {}),
        )


def _json(obj) -> str:
    return json.dumps(obj, default=str)


# --- per-user state ---

async def _load_settings(user_id: str) -> Optional[dict]:
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM agent_user_settings WHERE user_id = $1::uuid", user_id)
        return dict(row) if row else None


async def _load_paper_account(user_id: str) -> Optional[dict]:
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "SELECT id, api_key_id, api_secret_key_encrypted FROM alpaca_paper_accounts WHERE user_id = $1::uuid",
            user_id,
        )
        return dict(row) if row else None


async def _save_peak_and_latch(user_id: str, peak: float, latched: bool) -> None:
    async with service_conn() as conn:
        await conn.execute(
            """
            UPDATE agent_user_settings
            SET peak_equity = GREATEST(COALESCE(peak_equity, 0), $2),
                breaker_latched = $3,
                breaker_latched_at = CASE WHEN $3 AND NOT breaker_latched THEN now() ELSE breaker_latched_at END,
                updated_at = now()
            WHERE user_id = $1::uuid
            """,
            user_id, peak, latched,
        )


# --- market data ---

async def _latest_scores() -> dict[str, dict]:
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, as_of_date, short_score, short_signal, sector_key
            FROM stock_scores WHERE universe_id = $1
            ORDER BY ticker, as_of_date DESC
            """,
            UNIVERSE_ID,
        )
    if not rows:
        return {}
    newest = max(r["as_of_date"] for r in rows)
    return {r["ticker"]: dict(r) for r in rows if r["as_of_date"] == newest}


async def _history(ticker: str):
    return await run_in_threadpool(get_cached_history, ticker, "2y", True)


def _indicators(df) -> Optional[dict]:
    if df is None or df.empty or "Close" not in df:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    price = float(closes.iloc[-1])
    volumes = df["Volume"].reindex(closes.index).fillna(0) if "Volume" in df else None
    return {
        "price": price,
        "sma200": sma(closes, CONFIG.trend_sma_days),
        "avg_dollar_volume": average_dollar_volume(closes, volumes, CONFIG.dollar_volume_window_days) if volumes is not None else None,
        "volatility_pct": annualized_volatility_pct(closes, CONFIG.volatility_window_days),
        "atr14": atr(df["High"], df["Low"], df["Close"], 14) if {"High", "Low"} <= set(df.columns) else None,
        "returns": closes.pct_change().dropna().tail(CONFIG.volatility_window_days),
    }


def _trading_days_ahead_calendar_days(as_of: date, trading_days: int) -> int:
    """AGT-8: the calendar-day span covering the next `trading_days`
    trading sessions from as_of, skipping weekends -- exchange holidays
    aren't modelled (a known, disclosed simplification), so this can
    occasionally be one session short around a holiday, but it is no
    longer the flat 4-calendar-day window that was stricter than 2
    trading days on most weeks (Mon-Wed starts) and looser on others."""
    d = as_of
    counted = 0
    calendar_days = 0
    while counted < trading_days:
        d += timedelta(days=1)
        calendar_days += 1
        if d.weekday() < 5:  # Mon-Fri
            counted += 1
    return calendar_days


async def _earnings_blackout(ticker: str) -> bool:
    frame = await run_in_threadpool(get_cached_earnings_dates, ticker)
    today = datetime.now(EASTERN).date()
    window_days = _trading_days_ahead_calendar_days(today, CONFIG.earnings_blackout_trading_days)
    upcoming = upcoming_earnings_in_window(frame, as_of=today, window_days=window_days)
    return upcoming is not None


# --- broker helpers (sync calls run in a thread) ---

async def _broker(fn, creds, *args, **kwargs):
    return await run_in_threadpool(fn, creds["key"], creds["secret"], *args, **kwargs)


def _creds(account: dict) -> dict:
    return {"key": account["api_key_id"], "secret": decrypt_token(account["api_secret_key_encrypted"])}


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# --- stop integrity (AGT-16, AGT-29) ---

def _stop_trail_percent(price: float, atr14: Optional[float]) -> Optional[float]:
    if not atr14 or price <= 0:
        return None
    raw = CONFIG.stop_atr_multiple * atr14 / price * 100.0
    return round(min(max(raw, CONFIG.stop_min_pct), CONFIG.stop_max_pct), 2)


async def _ensure_stops(run_id: int, user_id: str, creds: dict, positions: list[dict], atr_by_ticker: dict[str, Optional[float]], prices: dict[str, float]) -> list[str]:
    """Returns the list of tickers still unprotected after this pass."""
    open_orders = await _broker(alpaca_trading_client.list_open_orders, creds)
    stops = {}
    for o in open_orders:
        if o.get("type") == "trailing_stop" and o.get("side") == "sell":
            stops.setdefault(o["symbol"], o)

    unprotected: list[str] = []
    for p in positions:
        ticker = p["symbol"]
        qty = math.floor(_f(p["qty"]))
        if qty < 1:
            continue
        existing = stops.get(ticker)
        if existing and math.floor(_f(existing.get("qty"))) == qty:
            continue

        trail = _stop_trail_percent(prices.get(ticker, _f(p.get("current_price"))), atr_by_ticker.get(ticker))
        if trail is None:
            unprotected.append(ticker)
            await _journal(run_id, user_id, "stop_failed", f"Could not size a stop for {ticker}: no ATR available.", ticker=ticker)
            continue

        if existing:
            try:
                await _broker(alpaca_trading_client.cancel_order, creds, existing["id"])
                await _journal(run_id, user_id, "stop_replaced", f"Position size of {ticker} changed; re-sizing its protective stop.",
                               ticker=ticker, qty=qty, alpaca_order_id=existing["id"])
            except AlpacaTradingError as e:
                unprotected.append(ticker)
                await _journal(run_id, user_id, "stop_failed", f"Could not cancel the old stop for {ticker}: {e.message}", ticker=ticker)
                continue

        cid = str(uuid.uuid4())
        try:
            placed = await _broker(alpaca_trading_client.submit_trailing_stop, creds,
                                   client_order_id=cid, ticker=ticker, qty=qty, trail_percent=trail)
            await _journal(run_id, user_id, "stop_placed",
                           f"Broker-side trailing stop at {trail:g}% (2.5 x ATR14, limited to 3-15%) protects {qty} shares of {ticker}.",
                           ticker=ticker, qty=qty, alpaca_order_id=placed.get("id"), client_order_id=cid,
                           detail={"trail_percent": trail})
        except (AlpacaTradingError, httpx.HTTPError) as e:
            unprotected.append(ticker)
            await _journal(run_id, user_id, "stop_failed", f"Could not place a stop for {ticker}: {e}", ticker=ticker, client_order_id=cid)
    return unprotected


# --- fills (AGT-25/28) ---

def _reconcile_outcome(event_type: str, ticker: str, side: Optional[str], order: dict) -> Optional[dict]:
    """Pure: classifies one order's current broker status into what to
    journal/alert, given which agent_order_events row it came from --
    'submitted' for the agent's own rebalance order, 'stop_placed' for a
    broker-side protective stop (AGT-16/29). Returns None if the order
    isn't terminal yet (still pending/open), same as the caller's own
    prior inline logic, just extracted so it can be tested without a DB."""
    status = order.get("status")
    is_stop = event_type == "stop_placed"
    if status == "filled":
        filled_qty = order.get("filled_qty")
        filled_price = _f(order.get("filled_avg_price"))
        if is_stop:
            reason = f"Protective stop triggered: sold {filled_qty} {ticker} at about ${filled_price:,.2f}."
            return {
                "journal_event": "filled", "side": "sell", "trigger": "stop_triggered",
                "alert_type": "agent_stop_triggered", "alert_subject": f"Stop triggered: {ticker}", "reason": reason,
            }
        reason = f"Filled {filled_qty} {ticker} at about ${filled_price:,.2f}."
        return {
            "journal_event": "filled", "side": side, "trigger": None,
            "alert_type": "agent_fill", "alert_subject": f"Agent {side} filled: {ticker}", "reason": reason,
        }
    if status in ("canceled", "rejected", "expired"):
        return {
            "journal_event": status, "side": side or "sell", "trigger": None,
            "alert_type": None, "alert_subject": None, "reason": f"Order ended as {status}.",
        }
    return None


async def _reconcile_submitted(run_id: int, user_id: str, creds: dict) -> None:
    """AGT-25/28. Covers both the agent's own rebalance orders
    (event_type='submitted') and its broker-side protective stops
    (event_type='stop_placed', from _ensure_stops) -- a triggered stop
    used to only be discovered implicitly, the next time _ensure_stops
    happened to notice the position was gone; it is now reconciled and
    alerted the same run-to-run cadence as any other fill, not just
    inferred later."""
    async with service_conn() as conn:
        pending = await conn.fetch(
            """
            SELECT e.alpaca_order_id, e.ticker, e.side, e.qty, e.event_type FROM agent_order_events e
            WHERE e.user_id = $1::uuid AND e.event_type = ANY($2::text[]) AND e.alpaca_order_id IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM agent_order_events t
                WHERE t.alpaca_order_id = e.alpaca_order_id AND t.event_type = ANY($3::text[])
              )
            """,
            user_id, ["submitted", "stop_placed"], list(TERMINAL_EVENTS),
        )
    for row in pending:
        try:
            order = await _broker(alpaca_trading_client.get_order, creds, row["alpaca_order_id"])
        except (AlpacaTradingError, httpx.HTTPError) as e:
            logger.warning("Agent reconcile failed for %s: %s", row["alpaca_order_id"], e)
            continue
        outcome = _reconcile_outcome(row["event_type"], row["ticker"], row["side"], order)
        if outcome is None:
            continue
        await _journal(run_id, user_id, outcome["journal_event"], outcome["reason"], ticker=row["ticker"],
                       side=outcome["side"], qty=_f(order.get("filled_qty")) if outcome["journal_event"] == "filled" else None,
                       alpaca_order_id=row["alpaca_order_id"], trigger=outcome["trigger"],
                       detail={"filled_avg_price": order.get("filled_avg_price")} if outcome["journal_event"] == "filled" else {})
        if outcome["alert_type"]:
            await dispatch_alert(user_id, row["ticker"], outcome["alert_type"], outcome["alert_subject"], outcome["reason"])


# --- the run ---

async def run_agent_for_user(user_id: str, *, trigger: str = "scheduled", force_plan: bool = False) -> dict:
    settings = await _load_settings(user_id)
    if not settings or not settings["enabled"]:
        return {"status": "not_enabled"}

    if force_plan:
        settings = {**settings, "mode": "plan"}
    mode = settings["mode"]
    kill_engaged = settings.get("kill_engaged") or await get_setting_bool(
        AGENT_KILL_SWITCH_KEY, default=False
    ) or await get_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, default=False)
    if kill_engaged:
        run_id = await _write_run_header(user_id, mode, "skipped", "Kill switch is engaged; no orders placed, existing stops left in place.", {})
        await _journal(run_id, user_id, "run_skipped", "Kill switch is engaged. No orders were placed; existing stops remain.")
        return {"status": "skipped", "reason": "kill_switch"}

    account_row = await _load_paper_account(user_id)
    if account_row is None:
        run_id = await _write_run_header(user_id, mode, "skipped", "No linked paper-trading account.", {})
        await _journal(run_id, user_id, "run_skipped", "No linked paper-trading account, so the agent has nothing to manage.")
        return {"status": "skipped", "reason": "no_account"}

    try:
        creds = _creds(account_row)
    except ValueError:
        await dispatch_alert(user_id, None, "agent_failure", "Trading agent cannot run: stored key unreadable",
                             "The trading agent could not decrypt your stored paper-trading key, so it placed nothing. Re-link the paper account.")
        return {"status": "failed", "error": "stored key unreadable"}

    try:
        return await _run_with_creds(user_id, settings, creds, account_row, trigger)
    except Exception as e:  # noqa: BLE001 -- every failure must be journaled and alerted, never swallowed
        logger.exception("Agent run failed for user %s", user_id)
        return await _handle_failed_run(user_id, settings, creds, str(e))


async def _handle_failed_run(user_id: str, settings: dict, creds: dict, error: str) -> dict:
    """AGT-29: a failed run still checks protection. Stops are re-verified
    (paper mode, kill switch off) so a crash mid-run never leaves a position
    naked. Anything still unprotected, or any failure in this step itself,
    alerts the user."""
    run_id = await _write_run_header(user_id, settings["mode"], "skipped", f"Run failed: {error}", {})
    await _journal(run_id, user_id, "run_failed", f"Run failed and was stopped: {error}")

    unprotected: list[str] = []
    protection_error: Optional[str] = None
    if settings["mode"] == "paper" and not settings.get("kill_engaged") and not await get_setting_bool(
        AGENT_KILL_SWITCH_KEY, default=False
    ):
        try:
            positions = await _broker(alpaca_trading_client.list_positions, creds)
            tickers = [p["symbol"] for p in positions]
            indicators = {t: _indicators(await _history(t)) or {} for t in tickers}
            atr_by_ticker = {t: indicators[t].get("atr14") for t in tickers}
            prices = {p["symbol"]: _f(p.get("current_price")) for p in positions}
            unprotected = await _ensure_stops(run_id, user_id, creds, positions, atr_by_ticker, prices)
        except Exception as e:  # noqa: BLE001
            protection_error = str(e)

    if protection_error or unprotected:
        detail = protection_error or f"positions without a stop: {', '.join(unprotected)}"
        await dispatch_alert(user_id, None, "agent_failure", "Trading agent run failed; protection needs attention",
                             f"The run failed ({error}). Stop check: {detail}. Check the Trading Agent page now.")
    else:
        await dispatch_alert(user_id, None, "agent_failure", "Trading agent run failed",
                             f"The run failed and was stopped: {error}. All open positions were checked and have protective stops.")
    return {"status": "failed", "error": error, "run_id": run_id, "unprotected": unprotected}


async def _run_with_creds(user_id: str, settings: dict, creds: dict, account_row: dict, trigger: str) -> dict:
    mode = settings["mode"]
    clock = await _broker(alpaca_trading_client.get_clock, creds)
    market_open = bool(clock.get("is_open"))
    account = await _broker(alpaca_trading_client.get_account, creds)
    equity = _f(account.get("equity"))
    last_equity = _f(account.get("last_equity"), default=equity)
    cash = _f(account.get("cash"))
    if equity <= 0:
        raise RuntimeError("Broker reports zero equity; refusing to plan.")

    regime = await regime_as_of()
    regime_cap, regime_reason = regime_cap_pct(regime)
    peak = max(settings.get("peak_equity") or 0.0, equity)
    latched = bool(settings.get("breaker_latched"))
    state = risk_state(equity, last_equity, peak, latched, regime_cap)
    if state.state == "circuit_breaker" and not latched:
        latched = True
        await dispatch_alert(user_id, None, "agent_risk_state", "Trading agent: drawdown circuit breaker tripped",
                             "Equity is 10% or more below its peak. Exposure is capped at 20% and buys are blocked until you reset the breaker.")
    await _save_peak_and_latch(user_id, peak, latched)

    snap = {"regime": regime, "exposure_cap_pct": state.exposure_cap_pct, "equity": equity,
            "last_equity": last_equity, "risk_state": state.state}
    run_id = await _write_run_header(user_id, mode, "started", f"Trigger: {trigger}. {regime_reason}", snap)

    scores = await _latest_scores()
    positions_raw = await _broker(alpaca_trading_client.list_positions, creds)
    held_tickers = [p["symbol"] for p in positions_raw]

    buy_tickers = sorted(
        (t for t, s in scores.items() if s["short_signal"] == "Buy"),
        key=lambda t: -(scores[t]["short_score"] or 0),
    )[: CONFIG.max_candidates_scanned]
    to_scan = list(dict.fromkeys(buy_tickers + held_tickers))

    indicators: dict[str, dict] = {}
    for t in to_scan:
        ind = _indicators(await _history(t))
        if ind:
            indicators[t] = ind

    candidates: list[Candidate] = []
    for t in buy_tickers:
        ind = indicators.get(t)
        if ind is None:
            candidates.append(Candidate(t, scores[t]["sector_key"] or "Unknown", 0.0, "Buy", None, None, None, False, scores[t]["short_score"]))
            continue
        candidates.append(Candidate(
            ticker=t, sector=scores[t]["sector_key"] or "Unknown", price=ind["price"], signal="Buy",
            sma200=ind["sma200"], avg_dollar_volume=ind["avg_dollar_volume"], volatility_pct=ind["volatility_pct"],
            earnings_blackout=await _earnings_blackout(t), short_score=scores[t]["short_score"],
        ))

    passed, rejected = filter_candidates(candidates)
    for t, reason in rejected:
        await _journal(run_id, user_id, "skipped", reason, ticker=t, side="buy")

    sized = size_positions(passed, equity, state.exposure_cap_pct)
    if sized:
        weights = {s["ticker"]: s["target_value"] / equity for s in sized}
        returns = _returns_frame({s["ticker"] for s in sized} | set(held_tickers), indicators)
        scale, est = scale_for_vol_target(weights, returns, CONFIG.vol_target_annual_pct)
        if scale < 1.0:
            for s in sized:
                s["target_value"] = round(s["target_value"] * scale, 2)
                s["weight_pct"] = round(s["target_value"] / equity * 100.0, 4)
        await _journal(run_id, user_id, "plan_note",
                       f"Estimated portfolio volatility {est}% vs the {CONFIG.vol_target_annual_pct:g}% target"
                       + (f"; exposure scaled by {scale:.2f}." if scale < 1.0 else "; no scaling needed."),
                       detail={"estimated_vol_pct": est, "scale": scale})

    holdings = []
    scored_held = {t: scores.get(t) for t in held_tickers}
    for p in positions_raw:
        t = p["symbol"]
        ind = indicators.get(t, {})
        price = _f(p.get("current_price"), default=ind.get("price", 0.0))
        holdings.append(Holding(
            ticker=t, sector=(scored_held[t] or {}).get("sector_key") or "Unknown",
            shares=_f(p.get("qty")), price=price,
            signal=(scored_held[t] or {}).get("short_signal"), sma200=ind.get("sma200"),
        ))

    orders, skipped = plan_orders(holdings, sized, equity, state)
    for t, reason in skipped:
        await _journal(run_id, user_id, "skipped", reason, ticker=t)

    sector_by_ticker = {h.ticker: h.sector for h in holdings}
    sector_by_ticker.update({s["ticker"]: s["sector"] for s in sized})
    for o in orders:
        await _journal(run_id, user_id, "proposed", o.reason, ticker=o.ticker, side=o.side, qty=o.qty,
                       est_price=o.est_price, est_value=o.est_value, trigger=o.trigger)

    if mode == "live":
        for o in orders:
            await _journal(run_id, user_id, "rejected", LIVE_BLOCK_REASON, ticker=o.ticker, side=o.side, qty=o.qty, trigger=o.trigger)
        await _journal(run_id, user_id, "run_completed", "Plan computed; live orders blocked by the validation gate.")
        return {"status": "completed", "run_id": run_id, "orders_proposed": len(orders), "live_blocked": True}

    if mode == "plan" or not market_open:
        note = "Plan mode: orders were computed and journaled, none were placed." if mode == "plan" else \
            "Market is closed (broker clock): plan journaled, no orders placed (AGT-4)."
        await _journal(run_id, user_id, "run_completed", note)
        return {"status": "completed", "run_id": run_id, "orders_proposed": len(orders), "orders_placed": 0}

    await _reconcile_submitted(run_id, user_id, creds)
    submitted = await _submit_orders(run_id, user_id, creds, orders, holdings, cash, equity, state, sector_by_ticker, indicators)

    if submitted:
        await _wait_for_fills(submitted, creds)

    positions_now = await _broker(alpaca_trading_client.list_positions, creds)
    prices = {p["symbol"]: _f(p.get("current_price")) for p in positions_now}
    atr_by_ticker = {t: indicators[t]["atr14"] for t in indicators}
    unprotected = await _ensure_stops(run_id, user_id, creds, positions_now, atr_by_ticker, prices)
    if unprotected:
        await dispatch_alert(user_id, None, "agent_failure", "Trading agent: positions missing a protective stop",
                             f"These positions do not yet have a broker-side stop: {', '.join(unprotected)}. Check the Trading Agent page.")

    await _journal(run_id, user_id, "run_completed", f"Run finished: {len(submitted)} order(s) submitted.")
    return {"status": "completed", "run_id": run_id, "orders_placed": len(submitted), "unprotected": unprotected}


async def _kill_engaged_now(user_id: str) -> bool:
    """Re-read every kill control right before each submit (AGT-3), so an
    engagement made mid-run takes effect on the next order, not the next run."""
    if await get_setting_bool(AGENT_KILL_SWITCH_KEY, default=False):
        return True
    if await get_setting_bool(PAPER_TRADING_KILL_SWITCH_KEY, default=False):
        return True
    settings = await _load_settings(user_id)
    return bool(settings and settings.get("kill_engaged"))


def _returns_frame(tickers: set[str], indicators: dict[str, dict]):
    import pandas as pd
    cols = {t: indicators[t]["returns"] for t in tickers if t in indicators and len(indicators[t].get("returns", [])) > 10}
    return pd.DataFrame(cols)


async def _submit_orders(run_id, user_id, creds, orders: list[Order], holdings: list[Holding], cash: float,
                         equity: float, state, sector_by_ticker: dict, indicators: dict) -> list[dict]:
    virtual = [Holding(h.ticker, h.sector, h.shares, h.price, h.signal, h.sma200) for h in holdings]
    running_cash = cash
    submitted: list[dict] = []

    for o in orders:
        kill_now = await _kill_engaged_now(user_id)
        if kill_now:
            await _journal(run_id, user_id, "rejected", "Kill switch engaged before this order; not sent.", ticker=o.ticker, side=o.side, qty=o.qty)
            continue

        reason = preflight(
            o, holdings=virtual, cash=running_cash, equity=equity, state=state,
            exposure_cap_pct=state.exposure_cap_pct, sector_by_ticker=sector_by_ticker,
        )
        if reason:
            await _journal(run_id, user_id, "rejected", reason, ticker=o.ticker, side=o.side, qty=o.qty, trigger=o.trigger)
            continue

        if o.side == "sell":
            await _cancel_existing_stop(run_id, user_id, creds, o.ticker)

        cid = str(uuid.uuid4())
        await _journal(run_id, user_id, "submitting", o.reason, ticker=o.ticker, side=o.side, qty=o.qty,
                       est_price=o.est_price, est_value=o.est_value, trigger=o.trigger, client_order_id=cid)
        try:
            placed = await _broker(alpaca_trading_client.submit_order, creds, client_order_id=cid, ticker=o.ticker,
                                   side=o.side, qty=o.qty, order_type="market", time_in_force="day")
        except AlpacaTradingError as e:
            await _journal(run_id, user_id, "rejected", f"Alpaca rejected the order: {e.message}", ticker=o.ticker,
                           side=o.side, qty=o.qty, client_order_id=cid)
            continue
        except httpx.HTTPError as e:
            recovered = await _broker(alpaca_trading_client.get_order_by_client_order_id, creds, cid)
            if recovered is None:
                await _journal(run_id, user_id, "rejected", f"Network failure before Alpaca accepted the order: {e}",
                               ticker=o.ticker, side=o.side, qty=o.qty, client_order_id=cid)
                continue
            placed = recovered

        await _journal(run_id, user_id, "submitted", o.reason, ticker=o.ticker, side=o.side, qty=o.qty,
                       alpaca_order_id=placed.get("id"), client_order_id=cid, trigger=o.trigger)
        submitted.append({"ticker": o.ticker, "side": o.side, "alpaca_order_id": placed.get("id")})

        if o.side == "sell":
            running_cash += o.est_value
            virtual = _apply_fill(virtual, o.ticker, -o.qty, o.est_price)
        else:
            running_cash -= o.est_value
            virtual = _apply_fill(virtual, o.ticker, o.qty, o.est_price, sector=sector_by_ticker.get(o.ticker, "Unknown"))
    return submitted


def _apply_fill(holdings: list[Holding], ticker: str, delta_shares: float, price: float, sector: str = "Unknown") -> list[Holding]:
    out = []
    found = False
    for h in holdings:
        if h.ticker == ticker:
            found = True
            new_shares = h.shares + delta_shares
            if new_shares > 1e-9:
                out.append(Holding(h.ticker, h.sector, new_shares, price, h.signal, h.sma200))
        else:
            out.append(h)
    if not found and delta_shares > 0:
        out.append(Holding(ticker, sector, delta_shares, price, None, None))
    return out


async def _cancel_existing_stop(run_id: int, user_id: str, creds: dict, ticker: str) -> None:
    """A sell changes the position size, so the stop protecting it is
    re-sized (AGT-16). Cancel first, otherwise Alpaca holds those shares
    for the stop and the sell is rejected."""
    open_orders = await _broker(alpaca_trading_client.list_open_orders, creds)
    for o in open_orders:
        if o.get("symbol") == ticker and o.get("type") == "trailing_stop" and o.get("side") == "sell":
            try:
                await _broker(alpaca_trading_client.cancel_order, creds, o["id"])
                await _journal(run_id, user_id, "stop_cancelled", f"Cancelled the trailing stop on {ticker} before selling into it.",
                               ticker=ticker, alpaca_order_id=o["id"])
            except AlpacaTradingError as e:
                logger.warning("Could not cancel stop %s: %s", o["id"], e.message)


async def _wait_for_fills(submitted: list[dict], creds: dict) -> None:
    """Market orders fill within seconds during the session. Wait briefly so
    fresh positions get their stops in the same run (AGT-29)."""
    deadline = time.monotonic() + FILL_WAIT_SECONDS
    pending = {s["alpaca_order_id"] for s in submitted if s.get("alpaca_order_id")}
    while pending and time.monotonic() < deadline:
        for oid in list(pending):
            try:
                order = await _broker(alpaca_trading_client.get_order, creds, oid)
            except (AlpacaTradingError, httpx.HTTPError):
                continue
            if order.get("status") in ("filled", "canceled", "rejected", "expired"):
                pending.discard(oid)
        if pending:
            await asyncio.sleep(FILL_POLL_SECONDS)
