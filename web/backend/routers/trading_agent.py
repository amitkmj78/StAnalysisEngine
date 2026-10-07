"""Phase 4 automated trading agent: user controls, read-only status/journal,
and admin-only per-user enablement (AGT-34/35).

Nothing here places an order directly. Orders come only from
services/agent/runner.py, and only in paper mode. Live mode is blocked at
the runner, so no endpoint here can enable it.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from starlette.concurrency import run_in_threadpool

import json

from services import alpaca_trading_client
from services.agent.config import CONFIG, config_version
from services.agent.runner import LIVE_BLOCK_REASON, run_agent_for_user
from services.agent.risk import regime_cap_pct
from services.agent.validation import (
    AGT30_BACKTEST_VALIDATION_SETTING_KEY,
    PAPER_TRADING_MIN_DAYS,
    live_trading_gate_status,
    run_agt30_stop_validation,
)
from services.alpaca_trading_client import AlpacaTradingError
from services.backtest_engine import max_drawdown_pct, sharpe
from services.challenge_service import compute_member_performance
from services.market_regime_service import REGIME_GATE_DISCLOSURE, regime_as_of
from services.yfinance_cache import get_cached_history
from web.backend.admin import require_admin
from web.backend.app_settings import (
    AGENT_ENABLED_KEY,
    AGENT_KILL_SWITCH_KEY,
    get_setting_bool,
    set_setting_bool,
    set_setting_str,
)
from web.backend.auth import verify_bearer_token
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

logger = logging.getLogger(__name__)

user_router = APIRouter(prefix="/api/v1/trading-agent", tags=["trading-agent"], dependencies=[Depends(verify_bearer_token)])
admin_router = APIRouter(prefix="/api/v1/admin/trading-agent", tags=["trading-agent-admin"], dependencies=[Depends(require_admin)])

AGENT_DISCLOSURE = "Automated trading can lose money. Past performance doesn't predict future results."


# AGT-2(b): the exact phrase a request must type to count as "explicit
# user opt-in" for live mode -- a plain mode="live" alone is not consent.
LIVE_TRADING_CONFIRMATION_PHRASE = "I UNDERSTAND THE RISK AND WANT LIVE TRADING"


class ModeRequest(BaseModel):
    mode: str = Field(pattern="^(plan|paper|live)$")
    live_confirmation_phrase: Optional[str] = None


class KillRequest(BaseModel):
    engaged: bool


class EnableUserRequest(BaseModel):
    email: EmailStr
    compliance_reference: str = Field(min_length=3, max_length=200)


class DisableUserRequest(BaseModel):
    email: EmailStr


class GlobalRequest(BaseModel):
    enabled: bool


async def _settings_for(user_id: str) -> Optional[dict]:
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT * FROM agent_user_settings WHERE user_id = $1::uuid", user_id)
    return dict(row) if row else None


async def _require_enabled(user_id: str) -> dict:
    settings = await _settings_for(user_id)
    if not settings or not settings["enabled"]:
        raise HTTPException(403, "The trading agent isn't enabled for your account yet. An admin must enable it after compliance review.")
    return settings


async def _live_positions(user_id: str) -> Optional[dict]:
    async with service_conn() as conn:
        account = await conn.fetchrow(
            "SELECT api_key_id, api_secret_key_encrypted FROM alpaca_paper_accounts WHERE user_id = $1::uuid", user_id
        )
    if account is None:
        return None
    try:
        creds = {"key": account["api_key_id"], "secret": decrypt_token(account["api_secret_key_encrypted"])}
        positions = await run_in_threadpool(alpaca_trading_client.list_positions, creds["key"], creds["secret"])
        orders = await run_in_threadpool(alpaca_trading_client.list_open_orders, creds["key"], creds["secret"])
    except (AlpacaTradingError, ValueError) as e:
        return {"error": str(e)}
    stops = {o["symbol"]: {"qty": o.get("qty"), "trail_percent": o.get("trail_percent"), "order_id": o.get("id")}
             for o in orders if o.get("type") == "trailing_stop" and o.get("side") == "sell"}
    return {
        "positions": [
            {"ticker": p["symbol"], "qty": float(p["qty"]), "market_value": float(p.get("market_value") or 0),
             "stop": stops.get(p["symbol"])}
            for p in positions
        ]
    }


def _worst_month(snaps: list[dict]) -> Optional[dict]:
    """AGT-27: groups the account's own daily equity snapshots by calendar
    month and returns the worst finished one (first vs last snapshot in
    that month) -- None while every month on record is still in progress,
    rather than reporting today's partial month as if it were finished."""
    by_month: dict[str, list[dict]] = {}
    for s in snaps:
        by_month.setdefault(s["as_of_date"].strftime("%Y-%m"), []).append(s)

    this_month = date.today().strftime("%Y-%m")
    worst = None
    for month, rows in by_month.items():
        if month == this_month:
            continue
        rows = sorted(rows, key=lambda r: r["as_of_date"])
        if len(rows) < 2:
            continue
        start_equity, end_equity = float(rows[0]["equity"]), float(rows[-1]["equity"])
        if start_equity <= 0:
            continue
        return_pct = round((end_equity / start_equity - 1.0) * 100.0, 2)
        if worst is None or return_pct < worst["return_pct"]:
            worst = {"month": month, "return_pct": return_pct}
    return worst


async def _performance(user_id: str) -> Optional[dict]:
    """AGT-27, paper only. Return, volatility, drawdown and Sharpe on the
    account's own equity snapshots, against SPY over the same window."""
    today = date.today()
    start = today - timedelta(days=90)
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT s.as_of_date, s.equity FROM paper_account_equity_snapshots s
            JOIN alpaca_paper_accounts a ON a.id = s.alpaca_paper_account_id
            WHERE a.user_id = $1::uuid AND s.as_of_date BETWEEN $2 AND $3
            ORDER BY s.as_of_date
            """,
            user_id, start, today,
        )
        fills = await conn.fetchval(
            "SELECT count(*) FROM agent_order_events WHERE user_id = $1::uuid AND event_type = 'filled' "
            "AND created_at >= $2",
            user_id, start,
        )
    if len(rows) < 2:
        return {"days_of_data": len(rows), "note": "Not enough equity history yet. Snapshots start once the paper equity capture job runs."}
    snaps = [{"as_of_date": r["as_of_date"], "equity": r["equity"]} for r in rows]
    perf = compute_member_performance(snaps, start, today)
    daily = [(snaps[i]["equity"] / snaps[i - 1]["equity"] - 1.0) * 100.0 for i in range(1, len(snaps))]
    spy_return = None
    try:
        spy = await run_in_threadpool(get_cached_history, "SPY", "1y", True)
        window = spy["Close"].loc[str(start):]
        if len(window) > 1:
            spy_return = round((float(window.iloc[-1]) / float(window.iloc[0]) - 1.0) * 100.0, 2)
    except Exception:  # noqa: BLE001 -- SPY is comparison context; never fail the page over it
        logger.warning("SPY comparison unavailable for agent performance")
    worst = _worst_month(snaps)
    return {
        "days_of_data": perf["days_of_data"],
        "return_pct": perf["return_pct"],
        "annualized_volatility_pct": perf["annualized_volatility_pct"],
        "max_drawdown_pct": max_drawdown_pct(daily),
        "sharpe": sharpe(daily, 0.0, 252),
        "spy_return_pct": spy_return,
        "costs_paid": f"{fills} order(s) filled in this window; $0 in commissions (Alpaca paper trading charges none).",
        "worst_month": (
            f"{worst['month']}: {worst['return_pct']:+.1f}%" if worst
            else "Not computed yet (needs at least one full finished calendar month of snapshots)."
        ),
    }


@user_router.get("/status")
async def get_status(request: Request):
    user_id = request.state.user["id"]
    settings = await _settings_for(user_id)
    regime = await regime_as_of()
    regime_cap, regime_reason = regime_cap_pct(regime)
    global_enabled = await get_setting_bool(AGENT_ENABLED_KEY, default=False)
    global_kill = await get_setting_bool(AGENT_KILL_SWITCH_KEY, default=False)

    latest_run = None
    if settings:
        async with service_conn() as conn:
            run = await conn.fetchrow(
                "SELECT * FROM agent_runs WHERE user_id = $1::uuid ORDER BY created_at DESC LIMIT 1", user_id
            )
            if run is not None:
                events = await conn.fetch(
                    "SELECT event_type, ticker, side, qty, est_value, trigger, reason, created_at "
                    "FROM agent_order_events WHERE run_id = $1 ORDER BY created_at, id",
                    run["id"],
                )
                latest_run = {"run": dict(run), "events": [dict(e) for e in events]}

    broker = await _live_positions(user_id) if settings and settings["enabled"] else None
    live_gate = await live_trading_gate_status(user_id)
    return {
        "enabled": bool(settings and settings["enabled"]),
        "mode": settings["mode"] if settings else "plan",
        "kill_engaged": bool(settings and settings["kill_engaged"]),
        "global_enabled": global_enabled,
        "global_kill_engaged": global_kill,
        "breaker_latched": bool(settings and settings["breaker_latched"]),
        "peak_equity": settings["peak_equity"] if settings else None,
        "config_version": config_version(),
        "limits": {
            "max_position_pct": CONFIG.max_position_pct,
            "max_sector_pct": CONFIG.max_sector_pct,
            "max_positions": CONFIG.max_positions,
            "daily_loss_limit_pct": CONFIG.daily_loss_limit_pct,
            "drawdown_breaker_pct": CONFIG.drawdown_breaker_pct,
        },
        "regime": {"label": regime, "exposure_cap_pct": regime_cap, "reason": regime_reason,
                   "disclosure": REGIME_GATE_DISCLOSURE},
        # AGT-2: the three gates, checked individually -- allowed is only
        # ever True once all three actually pass. services/agent/runner.py
        # also blocks live unconditionally underneath this, regardless of
        # what this reports.
        "live": {
            "allowed": live_gate.passed,
            "allow_live_trading_flag_set": live_gate.allow_live_trading_flag_set,
            "backtest_validation_passed": live_gate.backtest_validation_passed,
            "paper_trading_days": live_gate.paper_trading_days,
            "paper_trading_days_required": PAPER_TRADING_MIN_DAYS,
            "paper_trading_meets_bar": live_gate.paper_trading_meets_bar,
            "reasons": live_gate.reasons,
        },
        "disclosure": AGENT_DISCLOSURE,
        "broker": broker,
        "performance_paper": await _performance(user_id) if settings and settings["enabled"] else None,
        "latest_run": latest_run,
    }


@user_router.get("/journal")
async def get_journal(request: Request, limit: int = 20):
    user_id = request.state.user["id"]
    limit = max(1, min(limit, 100))
    async with service_conn() as conn:
        runs = await conn.fetch(
            "SELECT id, mode, status, regime, exposure_cap_pct, equity, risk_state, config_version, reason, created_at "
            "FROM agent_runs WHERE user_id = $1::uuid ORDER BY created_at DESC LIMIT $2",
            user_id, limit,
        )
        events = await conn.fetch(
            "SELECT run_id, event_type, ticker, side, qty, est_value, trigger, reason, alpaca_order_id, created_at "
            "FROM agent_order_events WHERE user_id = $1::uuid AND run_id = ANY($2::bigint[]) ORDER BY created_at, id",
            user_id, [r["id"] for r in runs] or [0],
        )
    by_run: dict[int, list] = {}
    for e in events:
        by_run.setdefault(e["run_id"], []).append({k: v for k, v in dict(e).items() if k != "run_id"})
    return {"runs": [{**dict(r), "events": by_run.get(r["id"], [])} for r in runs]}


@user_router.post("/run-plan")
@limiter.limit("10/minute")
async def run_plan_now(request: Request):
    """Preview: computes and journals the plan. Places no orders, whatever
    the saved mode is."""
    await enforce_daily_quota(request, "trading-agent/run-plan")
    user_id = request.state.user["id"]
    await _require_enabled(user_id)
    return await run_agent_for_user(user_id, trigger="manual_plan_preview", force_plan=True)


@user_router.post("/mode")
async def set_mode(request: Request, body: ModeRequest):
    user_id = request.state.user["id"]
    await _require_enabled(user_id)
    if body.mode == "live":
        # AGT-2: each of the three gates is checked and named individually
        # here, rather than one unconditional block -- but the outcome is
        # the same today regardless, since services/agent/runner.py also
        # blocks live unconditionally underneath this.
        gate = await live_trading_gate_status(user_id)
        reasons = list(gate.reasons)
        if body.live_confirmation_phrase != LIVE_TRADING_CONFIRMATION_PHRASE:
            reasons.append(
                f'Typed confirmation required: send live_confirmation_phrase exactly as "{LIVE_TRADING_CONFIRMATION_PHRASE}".'
            )
        reason_text = LIVE_BLOCK_REASON if not reasons else "Live trading is blocked: " + " ".join(reasons)
        async with service_conn() as conn:
            run_id = await conn.fetchval(
                """
                INSERT INTO agent_runs (user_id, mode, status, config_version, reason)
                VALUES ($1::uuid, 'live', 'skipped', $2, $3) RETURNING id
                """,
                user_id, config_version(), reason_text,
            )
            await conn.execute(
                "INSERT INTO agent_order_events (run_id, user_id, event_type, reason) VALUES ($1, $2::uuid, 'rejected', $3)",
                run_id, user_id, reason_text,
            )
        raise HTTPException(400, reason_text)
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE agent_user_settings SET mode = $2, updated_at = now() WHERE user_id = $1::uuid", user_id, body.mode
        )
    return {"mode": body.mode}


@user_router.post("/kill")
async def set_kill(request: Request, body: KillRequest):
    """Per-user kill (AGT-3). Stops new orders on the next check; existing
    protective stops are left in place."""
    user_id = request.state.user["id"]
    await _require_enabled(user_id)
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE agent_user_settings SET kill_engaged = $2, updated_at = now() WHERE user_id = $1::uuid",
            user_id, body.engaged,
        )
    return {"kill_engaged": body.engaged}


@user_router.post("/reset-breaker")
async def reset_breaker(request: Request):
    """AGT-18: the only way out of the drawdown circuit breaker. Journaled."""
    user_id = request.state.user["id"]
    await _require_enabled(user_id)
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE agent_user_settings SET breaker_latched = false, updated_at = now() WHERE user_id = $1::uuid", user_id
        )
        run_id = await conn.fetchval(
            """
            INSERT INTO agent_runs (user_id, mode, status, config_version, reason)
            SELECT user_id, mode, 'skipped', $2, 'Drawdown circuit breaker manually reset by the user.'
            FROM agent_user_settings WHERE user_id = $1::uuid RETURNING id
            """,
            user_id, config_version(),
        )
        await conn.execute(
            """
            INSERT INTO agent_order_events (run_id, user_id, event_type, reason)
            VALUES ($1, $2::uuid, 'breaker_reset', 'Drawdown circuit breaker reset manually. Normal sizing resumes on the next run.')
            """,
            run_id, user_id,
        )
    return {"breaker_latched": False}


# --- admin ---

@admin_router.post("/users/enable")
async def admin_enable_user(body: EnableUserRequest, request: Request):
    """AGT-34/35: enabling a user requires a non-empty compliance reference.
    Starts in plan mode, so enabling never places orders by itself."""
    admin_id = request.state.user["id"]
    async with service_conn() as conn:
        target = await conn.fetchval("SELECT id FROM users WHERE email = $1", body.email.lower())
        if target is None:
            raise HTTPException(404, "No account with that email.")
        await conn.execute(
            """
            INSERT INTO agent_user_settings (user_id, enabled, mode, compliance_reference, enabled_by, enabled_at, updated_at)
            VALUES ($1, true, 'plan', $2, $3::uuid, now(), now())
            ON CONFLICT (user_id) DO UPDATE SET enabled = true, mode = 'plan', compliance_reference = EXCLUDED.compliance_reference,
                enabled_by = EXCLUDED.enabled_by, enabled_at = now(), updated_at = now()
            """,
            target, body.compliance_reference.strip(), admin_id,
        )
    return {"ok": True, "email": body.email.lower(), "mode": "plan"}


@admin_router.post("/users/disable")
async def admin_disable_user(body: DisableUserRequest):
    async with service_conn() as conn:
        result = await conn.execute(
            "UPDATE agent_user_settings SET enabled = false, updated_at = now() "
            "WHERE user_id = (SELECT id FROM users WHERE email = $1)",
            body.email.lower(),
        )
    if result == "UPDATE 0":
        raise HTTPException(404, "That user has no agent settings.")
    return {"ok": True}


@admin_router.post("/global")
async def admin_set_global(body: GlobalRequest):
    await set_setting_bool(AGENT_ENABLED_KEY, body.enabled)
    return {"agent_enabled": body.enabled}


@admin_router.post("/kill-switch")
async def admin_set_kill(body: KillRequest):
    await set_setting_bool(AGENT_KILL_SWITCH_KEY, body.engaged)
    return {"agent_kill_switch": body.engaged}


@admin_router.post("/validate-agt30")
async def admin_validate_agt30():
    """Runs and persists the AGT-30 backtest leg (services/agent/
    validation.py) against SPY's real price history, for GET /status's
    live_trading_gate_status to read back. Not scheduled -- re-run by
    hand whenever the agent's stop-loss config changes; a stored result
    from under a different config_version is treated as stale, not as a
    current pass. See validation.py's own module docstring for exactly
    what this does and does not test (the regime cap is excluded, and
    this tests the stop-loss rule on SPY itself, not a multi-stock
    universe -- both disclosed in the result)."""
    spy = await run_in_threadpool(get_cached_history, "SPY", "max", True)
    if spy.empty:
        raise HTTPException(502, "No SPY history returned -- cannot run the validation.")
    report = run_agt30_stop_validation(spy)
    await set_setting_str(AGT30_BACKTEST_VALIDATION_SETTING_KEY, json.dumps(report))
    return report
