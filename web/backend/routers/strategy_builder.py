"""Strategy Builder: backtests, presets, saved strategies, comparison and read-only share links.

Nothing here places an order. Paper forward runs (STB-6) are not built.
"""

import hashlib
import json
import threading
import time
from datetime import date, timedelta
import random
import secrets
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.stock_finder_service import fetch_sp500_tickers, get_peer_lookup_table
from services.signal_publication_service import DEFAULT_HORIZON_DAYS, DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE, build_model_portfolio_series
from services.strategy_explainer import explain_backtest
from services.strategy_engine import DEFAULT_COOLDOWN, MAX_TICKERS, Rule, feature_frame, model_portfolio_summary, run_backtest
from services.sp500_membership import current_members, member_flags, members_on, removed_after
from services.strategy_scan import TEMPLATES, pick_sample, scan
from services.yfinance_cache import get_cached_history, get_earnings_report_dates
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/strategy-builder", tags=["strategy-builder"], dependencies=[Depends(verify_bearer_token)])
shared_router = APIRouter(prefix="/api/v1/strategy-builder", tags=["strategy-builder-shared"])

HISTORY_PERIOD = "5y"
HISTORY_DAYS = 365 * 5 + 2  # the backtest's five-year window
# SCAN-4: the scan uses ten years, so the walk-forward has 16 six-month test windows and covers 2018, 2020 and 2022.
SCAN_HISTORY_PERIOD = "10y"
SCAN_HISTORY_DAYS = 365 * 10 + 2
VARIANT_WINDOW_DAYS = 90
BENCHMARK = "SPY"
MAX_SAVED_RESULT_CHARS = 400_000
MAX_COMPARE = 4


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
    tickers: list[str] = Field(default_factory=list, max_length=MAX_TICKERS)
    entry: list[RuleIn] = Field(min_length=1, max_length=5)
    exit: list[RuleIn] = Field(default_factory=list, max_length=5)
    exits: ExitsIn = Field(default_factory=ExitsIn)
    waive_protective_exit: bool = False
    cooldown_sessions: int = Field(default=DEFAULT_COOLDOWN, ge=0, le=250)
    verdict_benchmark: Literal["basket", "spy"] = "basket"
    # Selection bias: "random_sample" draws the tickers from the S&P 500 with a seed, so the run can be repeated.
    source: Literal["hand_picked", "random_sample", "portfolio"] = "hand_picked"
    # Actual weights for a portfolio test (ticker -> share of value). Must cover exactly the tickers.
    weights: Optional[dict[str, float]] = None
    sample_seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    sample_size: Optional[int] = Field(default=None, ge=1, le=MAX_TICKERS)


def _definition_hash(tickers: list[str], body: BacktestRequest, weights: Optional[dict[str, float]] = None) -> str:
    canonical = json.dumps(
        {
            "tickers": sorted(set(tickers)),
            "entry": [r.model_dump() for r in body.entry],
            "exit": [r.model_dump() for r in body.exit],
            "exits": body.exits.model_dump(),
            "waive": body.waive_protective_exit,
            "cooldown": body.cooldown_sessions,
            "verdict": body.verdict_benchmark,
            "weights": sorted((weights or {}).items()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _sp500() -> list[str]:
    tickers = await run_in_threadpool(fetch_sp500_tickers)
    if not tickers:
        raise HTTPException(503, "The S&P 500 list is unavailable right now.")
    return sorted(tickers)


@router.post("/backtest")
@limiter.limit("10/minute")
async def backtest(request: Request, body: BacktestRequest):
    await enforce_daily_quota(request, "strategy-builder/backtest")
    user_id = request.state.user["id"]

    selection: dict = {"source": body.source, "seed": None}
    weights = None
    if body.weights:
        weights = {k.strip().upper(): float(v) for k, v in body.weights.items()}
    if body.source == "random_sample":
        size = body.sample_size or max(len(body.tickers), 5)
        seed = body.sample_seed if body.sample_seed is not None else random.randrange(1, 2_147_483_647)
        universe = await _sp500()
        tickers = random.Random(seed).sample(universe, min(size, len(universe)))
        selection = {"source": "random_sample", "seed": seed, "size": len(tickers)}
    else:
        tickers = [t.strip().upper() for t in body.tickers if t.strip()]
    if weights:
        tickers = list(weights)
        selection = {"source": "portfolio", "seed": None}
    tickers = sorted(set(tickers))
    if not tickers or any(not t.replace(".", "").replace("-", "").isalnum() for t in tickers):
        raise HTTPException(422, "Enter ticker symbols, for example AAPL, MSFT")
    if len(tickers) > MAX_TICKERS:
        raise HTTPException(422, f"At most {MAX_TICKERS} tickers.")

    entry_raw = [r.model_dump() for r in body.entry]
    exit_raw = [r.model_dump() for r in body.exit]
    try:
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
        reports = await run_in_threadpool(get_earnings_report_dates, t)
        frames[t] = feature_frame(history[columns], regime_by_date, reports)
    bench_hist = await run_in_threadpool(get_cached_history, BENCHMARK, HISTORY_PERIOD, True, None)
    if bench_hist.empty:
        raise HTTPException(503, "Benchmark price history is unavailable right now.")
    bench_close = bench_hist["Close"]

    definition = _definition_hash(tickers, body, weights)
    async with user_conn(user_id) as conn:
        prior = await conn.fetch(
            """
            SELECT definition_hash, sharpe_daily FROM strategy_backtest_runs
            WHERE user_id = $1::uuid AND created_at > now() - make_interval(days => $2)
            """,
            user_id, VARIANT_WINDOW_DAYS,
        )
    prior_sharpe: dict[str, float] = {}
    prior_hashes = set()
    for row in prior:
        prior_hashes.add(row["definition_hash"])
        if row["sharpe_daily"] is not None and row["definition_hash"] != definition:
            prior_sharpe[row["definition_hash"]] = float(row["sharpe_daily"])
    variants = len(prior_hashes | {definition})

    try:
        result = await run_in_threadpool(
            lambda: run_backtest(
                frames, entry_raw, exit_raw, bench_close, variants,
                exits_raw=body.exits.model_dump(),
                waive_protective_exit=body.waive_protective_exit,
                cooldown=body.cooldown_sessions,
                verdict_benchmark=body.verdict_benchmark,
                trial_sharpes_daily=list(prior_sharpe.values()),
                weights=weights,
            ),
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    async with user_conn(user_id) as conn:
        await conn.execute(
            "INSERT INTO strategy_backtest_runs (user_id, definition_hash, sharpe_daily) VALUES ($1::uuid, $2, $3)",
            user_id, definition, result.get("sharpe_daily"),
        )
    # STB-5: the app's own model portfolio, for comparison. Built from its saved outcomes, so it is withheld until it has history.
    async with service_conn() as conn:
        outcome_rows = await conn.fetch(
            "SELECT target_date, realized_return_pct FROM signal_outcomes WHERE universe_id = $1 AND lookback_days = $2 AND horizon_days = $3 ORDER BY target_date",
            DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS, DEFAULT_HORIZON_DAYS,
        )
    model_series = build_model_portfolio_series(
        [{"target_date": r["target_date"], "realized_return_pct": float(r["realized_return_pct"])} for r in outcome_rows],
        DEFAULT_HORIZON_DAYS,
    )
    result["model_portfolio"] = model_portfolio_summary(model_series)
    result["selection"] = {**selection, "tickers": tickers}
    # The plain-language report is built from this result's own numbers. If it can't be built, the result is still returned.
    try:
        result["explanation"] = explain_backtest(result, {
            "entry": [r.model_dump() for r in body.entry],
            "exit": [r.model_dump() for r in body.exit],
            "exits": body.exits.model_dump(),
        })
    except Exception:
        result["explanation"] = None
    result["disclaimer"] = (
        "Backtest of past prices with the rules and costs shown. Not a forecast, not a recommendation, "
        "and not an order: nothing is placed."
    )
    return result


@router.get("/presets")
@limiter.limit("30/minute")
async def presets(
    request: Request,
    kind: Literal["sp500_sample", "sector"],
    size: int = Query(10, ge=1, le=MAX_TICKERS),
    seed: Optional[int] = Query(None, ge=0, le=2_147_483_647),
    sector: Optional[str] = Query(None, max_length=60),
):
    """SB-U3: a seeded random sample of S&P 500 members, or the largest stocks in one GICS sector.
    The sample seed is returned so the same list can be reproduced."""
    if kind == "sp500_sample":
        universe = await _sp500()
        chosen_seed = seed if seed is not None else random.randrange(1, 2_147_483_647)
        tickers = random.Random(chosen_seed).sample(universe, min(size, len(universe)))
        return {"kind": kind, "tickers": sorted(tickers), "seed": chosen_seed}
    if not sector:
        raise HTTPException(422, "sector is required for a sector basket")
    table = await run_in_threadpool(get_peer_lookup_table, "All")
    if table.empty or "GICS Sector" not in table:
        raise HTTPException(503, "Sector data is unavailable right now.")
    in_sector = table[table["GICS Sector"] == sector].sort_values("Market Cap ($B)", ascending=False)
    tickers = in_sector["Ticker"].tolist()[:size]
    if not tickers:
        raise HTTPException(404, f"No stocks found for the sector {sector}.")
    return {"kind": kind, "sector": sector, "tickers": tickers}


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    definition: BacktestRequest
    result: dict


def _verdict_row(result: dict) -> dict:
    verdict = result.get("verdict") or {}
    strategy = result.get("strategy") or {}
    basket = result.get("basket") or {}
    spy = result.get("benchmark_spy") or {}
    checks = result.get("checks") or []
    return {
        "total_return_pct": strategy.get("total_return_pct"),
        "cagr_pct": strategy.get("cagr_pct"),
        "max_drawdown_pct": strategy.get("max_drawdown_pct"),
        "sharpe": strategy.get("sharpe"),
        "excess_cagr_vs_basket_pct": (strategy.get("cagr_pct") - basket["cagr_pct"]) if strategy.get("cagr_pct") is not None and basket.get("cagr_pct") is not None else None,
        "excess_cagr_vs_spy_pct": (strategy.get("cagr_pct") - spy["cagr_pct"]) if strategy.get("cagr_pct") is not None and spy.get("cagr_pct") is not None else None,
        "sharpe_vs_basket": verdict.get("sharpe_vs_basket"),
        "sharpe_vs_spy": verdict.get("sharpe_vs_spy"),
        "trades": result.get("trades"),
        "churn_pct": result.get("churn_pct"),
        "cost_drag_points": (result.get("cost_drag") or {}).get("cagr_points"),
        "deflated_probability": (result.get("deflated_sharpe") or {}).get("probability"),
        "checks_failed": sum(1 for c in checks if c.get("status") == "fail"),
        "checks_caution": sum(1 for c in checks if c.get("status") == "caution"),
    }


@router.post("/saved")
@limiter.limit("30/minute")
async def save_strategy(request: Request, body: SaveRequest):
    user_id = request.state.user["id"]
    result = {k: v for k, v in body.result.items() if k not in ("trade_log", "disclaimer")}
    payload = json.dumps(result, default=str)
    if len(payload) > MAX_SAVED_RESULT_CHARS:
        raise HTTPException(413, "This result is too large to save.")
    end_text = (result.get("period") or {}).get("end")
    data_end = date.fromisoformat(end_text) if end_text else None
    async with user_conn(user_id) as conn:
        new_id = await conn.fetchval(
            """
            INSERT INTO saved_strategies (user_id, name, definition, result, data_end)
            VALUES ($1::uuid, $2, $3::jsonb, $4::jsonb, $5::date)
            RETURNING id
            """,
            user_id, body.name.strip(), body.definition.model_dump_json(), payload, data_end,
        )
    return {"id": new_id}


@router.get("/saved")
@limiter.limit("60/minute")
async def list_saved(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT id, name, created_at, data_end, result FROM saved_strategies ORDER BY created_at DESC LIMIT 100"
        )
    return {
        "saved": [
            {"id": r["id"], "name": r["name"], "created_at": r["created_at"].isoformat(),
             "data_end": str(r["data_end"]) if r["data_end"] else None,
             "summary": _verdict_row(json.loads(r["result"]) if isinstance(r["result"], str) else r["result"])}
            for r in rows
        ]
    }


@router.get("/saved/{saved_id}")
@limiter.limit("60/minute")
async def get_saved(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "SELECT id, name, created_at, definition, result, share_token FROM saved_strategies WHERE id = $1", saved_id
        )
    if row is None:
        raise HTTPException(404, "Saved strategy not found.")
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"].isoformat(),
        "definition": json.loads(row["definition"]) if isinstance(row["definition"], str) else row["definition"],
        "result": json.loads(row["result"]) if isinstance(row["result"], str) else row["result"],
        "share_token": row["share_token"],
    }


@router.delete("/saved/{saved_id}")
@limiter.limit("30/minute")
async def delete_saved(request: Request, saved_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        status = await conn.execute("DELETE FROM saved_strategies WHERE id = $1", saved_id)
    if status.endswith(" 0"):
        raise HTTPException(404, "Saved strategy not found.")
    return {"ok": True}


class CompareRequest(BaseModel):
    ids: list[int] = Field(min_length=2, max_length=MAX_COMPARE)


@router.post("/compare")
@limiter.limit("30/minute")
async def compare(request: Request, body: CompareRequest):
    if len(set(body.ids)) != len(body.ids):
        raise HTTPException(422, "Pick each saved strategy once.")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(
            "SELECT id, name, result FROM saved_strategies WHERE id = ANY($1::bigint[])", body.ids
        )
    if len(rows) != len(body.ids):
        raise HTTPException(404, "One of the saved strategies was not found.")
    by_id = {r["id"]: r for r in rows}
    return {
        "rows": [
            {"id": i, "name": by_id[i]["name"], **_verdict_row(
                json.loads(by_id[i]["result"]) if isinstance(by_id[i]["result"], str) else by_id[i]["result"])}
            for i in body.ids
        ]
    }


@router.post("/saved/{saved_id}/share")
@limiter.limit("20/minute")
async def share_saved(request: Request, saved_id: int):
    """Creates a read-only link. Anyone with the link can view the result; the owner's identity is not shown."""
    user_id = request.state.user["id"]
    token = secrets.token_urlsafe(16)
    async with user_conn(user_id) as conn:
        status = await conn.execute(
            "UPDATE saved_strategies SET share_token = COALESCE(share_token, $2) WHERE id = $1", saved_id, token
        )
        if status.endswith(" 0"):
            raise HTTPException(404, "Saved strategy not found.")
        current = await conn.fetchval("SELECT share_token FROM saved_strategies WHERE id = $1", saved_id)
    return {"share_token": current, "path": f"/strategies/shared/{current}"}


@shared_router.get("/shared/{token}")
@limiter.limit("60/minute")
async def get_shared(request: Request, token: str):
    if not 10 <= len(token) <= 64:
        raise HTTPException(404, "Shared strategy not found.")
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "SELECT name, created_at, definition, result FROM saved_strategies WHERE share_token = $1", token
        )
    if row is None:
        raise HTTPException(404, "Shared strategy not found.")
    return {
        "name": row["name"],
        "created_at": row["created_at"].isoformat(),
        "definition": json.loads(row["definition"]) if isinstance(row["definition"], str) else row["definition"],
        "result": json.loads(row["result"]) if isinstance(row["result"], str) else row["result"],
        "read_only": True,
    }


# Template scan: runs in a background thread so the page can show progress. Jobs live in memory and
# are lost on a server restart, which is fine for a scan that takes a minute or two.
SCAN_JOBS: dict[str, dict] = {}
SCAN_KEEP = 20


def _run_scan_job(job_id: str, tickers: list[str], regime_by_date: Optional[dict[str, str]],
                  universe_info: dict, start_members: Optional[set]) -> None:
    job = SCAN_JOBS[job_id]
    try:
        frames = {}
        skipped = []
        for t in tickers:
            history = get_cached_history(t, SCAN_HISTORY_PERIOD, True, None)
            if history.empty or not {"Open", "High", "Low", "Close"}.issubset(history.columns):
                skipped.append(t)  # SCAN-2: a stock with no price history can't be tested; it is listed, not hidden
                continue
            columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in history.columns]
            flags = None
            if start_members is not None:
                flags = member_flags(t, history.index, start_members)
            frames[t] = feature_frame(history[columns], regime_by_date, get_earnings_report_dates(t), flags)
        bench = get_cached_history(BENCHMARK, SCAN_HISTORY_PERIOD, True, None)
        if len(frames) < 2 or bench.empty:
            raise ValueError("Not enough price history for this sample. Try again.")

        def progress(done: int, total: int) -> None:
            job["done"] = done

        job["result"] = scan(frames, bench["Close"], progress=progress)
        job["result"]["seed"] = job["seed"]
        job["result"]["universe"] = {**universe_info, "sampled": sorted(tickers), "skipped_no_prices": sorted(skipped),
                                     "tested": sorted(frames)}
        job["status"] = "done"
    except Exception as e:  # reported to the page, not raised in the thread
        job["status"] = "error"
        job["error"] = str(e) or "The scan failed."
    finally:
        job["finished"] = time.time()


@router.post("/scan")
@limiter.limit("3/minute")
async def start_scan(
    request: Request,
    seed: Optional[int] = Query(None, ge=0, le=2_147_483_647),
    current_only: bool = Query(False, description="SCAN-2: test today's members back in time (biased). Off by default."),
):
    await enforce_daily_quota(request, "strategy-builder/scan")
    chosen_seed = seed if seed is not None else random.randrange(1, 2_147_483_647)
    start = (date.today() - timedelta(days=SCAN_HISTORY_DAYS)).isoformat()
    if current_only:
        universe = await _sp500()
        start_members = None
        universe_info = {"basis": "current_members_biased", "as_of": start, "members_at_start": len(universe),
                         "left_index_in_window": []}
    else:
        start_members = await run_in_threadpool(members_on, start)
        universe = sorted(start_members)
        left = await run_in_threadpool(removed_after, start)
        universe_info = {"basis": "point_in_time", "as_of": start, "members_at_start": len(universe),
                         "left_index_in_window": left}
    if not universe:
        raise HTTPException(503, "The S&P 500 membership history is unavailable right now.")
    tickers = pick_sample(universe, chosen_seed)
    regime_by_date = None
    async with service_conn() as conn:
        rows = await conn.fetch("SELECT as_of_date, regime_confirmed FROM market_regime_daily")
    regime_by_date = {r["as_of_date"].isoformat(): r["regime_confirmed"] for r in rows}
    job_id = secrets.token_urlsafe(8)
    SCAN_JOBS[job_id] = {
        "status": "running", "done": 0, "total": len(TEMPLATES), "seed": chosen_seed,
        "tickers": tickers, "result": None, "error": None, "started": time.time(), "finished": None,
    }
    for old in sorted(SCAN_JOBS, key=lambda k: SCAN_JOBS[k]["started"])[:-SCAN_KEEP]:
        SCAN_JOBS.pop(old, None)
    threading.Thread(target=_run_scan_job, args=(job_id, tickers, regime_by_date, universe_info, start_members), daemon=True).start()
    return {"job_id": job_id, "seed": chosen_seed, "tickers": tickers, "universe": universe_info}


@router.get("/scan/{job_id}")
@limiter.limit("120/minute")
async def scan_status(request: Request, job_id: str):
    job = SCAN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "This scan is not found. It may have been cleared by a restart; start it again.")
    return {
        "status": job["status"],
        "done": job["done"],
        "total": job["total"],
        "seed": job["seed"],
        "tickers": job["tickers"],
        "error": job["error"],
        "result": job["result"],
    }
