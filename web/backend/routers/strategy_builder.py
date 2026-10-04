"""STB-1..5: no-code strategy backtests. Runs the pure engine in services/strategy_engine.py.

Nothing here places an order. STB-6 (paper forward runs) is not built.
"""

import hashlib
import json
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.strategy_engine import DEFAULT_COOLDOWN, MAX_TICKERS, Rule, feature_frame, run_backtest
from services.yfinance_cache import get_cached_history
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/strategy-builder", tags=["strategy-builder"], dependencies=[Depends(verify_bearer_token)])

HISTORY_PERIOD = "5y"
VARIANT_WINDOW_DAYS = 90
BENCHMARK = "SPY"


class RuleIn(BaseModel):
    field: str = Field(max_length=40)
    op: str = Field(max_length=20)
    value: Optional[str | float | int] = None


class ExitsIn(BaseModel):
    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    atr_stop_k: Optional[float] = None
    time_stop_sessions: Optional[int] = None
    take_profit_pct: Optional[float] = None


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=MAX_TICKERS)
    entry: list[RuleIn] = Field(min_length=1, max_length=5)
    exit: list[RuleIn] = Field(default_factory=list, max_length=5)
    exits: ExitsIn = Field(default_factory=ExitsIn)
    waive_protective_exit: bool = False
    cooldown_sessions: int = Field(default=DEFAULT_COOLDOWN, ge=0, le=250)
    verdict_benchmark: Literal["basket", "spy"] = "basket"


def _definition_hash(body: BacktestRequest) -> str:
    canonical = json.dumps(
        {
            "tickers": sorted({t.strip().upper() for t in body.tickers}),
            "entry": [r.model_dump() for r in body.entry],
            "exit": [r.model_dump() for r in body.exit],
            "exits": body.exits.model_dump(),
            "waive": body.waive_protective_exit,
            "cooldown": body.cooldown_sessions,
            "verdict": body.verdict_benchmark,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


@router.post("/backtest")
@limiter.limit("10/minute")
async def backtest(request: Request, body: BacktestRequest):
    await enforce_daily_quota(request, "strategy-builder/backtest")
    user_id = request.state.user["id"]
    tickers = sorted({t.strip().upper() for t in body.tickers if t.strip()})
    if not tickers or any(not t.replace(".", "").replace("-", "").isalnum() for t in tickers):
        raise HTTPException(422, "Enter ticker symbols, for example AAPL, MSFT")

    entry_raw = [r.model_dump() for r in body.entry]
    exit_raw = [r.model_dump() for r in body.exit]
    try:
        # Validate before logging, so only real attempts count toward the variant total.
        [Rule.parse(r) for r in entry_raw + exit_raw]
        if not body.waive_protective_exit and not (body.exit or body.exits.model_dump(exclude_none=True)):
            raise ValueError("add a protective exit (stop, trailing stop, ATR stop or time stop), or waive it")
    except ValueError as e:
        raise HTTPException(422, str(e))
    uses_regime = any(r["field"] == "regime" for r in entry_raw + exit_raw)

    regime_by_date: Optional[dict[str, str]] = None
    if uses_regime:
        async with service_conn() as conn:
            rows = await conn.fetch("SELECT as_of_date, regime_confirmed FROM market_regime_daily")
        regime_by_date = {r["as_of_date"].isoformat(): r["regime_confirmed"] for r in rows}

    frames = {}
    for t in tickers:
        history = await run_in_threadpool(get_cached_history, t, HISTORY_PERIOD, True, None)
        if history.empty or not {"Open", "Close"}.issubset(history.columns):
            raise HTTPException(422, f"No price history for {t}.")
        columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in history.columns]
        frames[t] = feature_frame(history[columns], regime_by_date)
    bench_hist = await run_in_threadpool(get_cached_history, BENCHMARK, HISTORY_PERIOD, True, None)
    if bench_hist.empty:
        raise HTTPException(503, "Benchmark price history is unavailable right now.")
    bench_close = bench_hist["Close"]

    definition = _definition_hash(body)
    async with user_conn(user_id) as conn:
        await conn.execute(
            "INSERT INTO strategy_backtest_runs (user_id, definition_hash) VALUES ($1::uuid, $2)",
            user_id, definition,
        )
        tried = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT definition_hash) FROM strategy_backtest_runs
            WHERE user_id = $1::uuid AND created_at > now() - make_interval(days => $2)
            """,
            user_id, VARIANT_WINDOW_DAYS,
        )

    try:
        result = await run_in_threadpool(
            lambda: run_backtest(
                frames, entry_raw, exit_raw, bench_close, int(tried or 1),
                exits_raw=body.exits.model_dump(),
                waive_protective_exit=body.waive_protective_exit,
                cooldown=body.cooldown_sessions,
                verdict_benchmark=body.verdict_benchmark,
            ),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    result["disclaimer"] = (
        "Backtest of past prices with the rules and costs shown. Not a forecast, not a recommendation, "
        "and not an order: nothing is placed."
    )
    return result
