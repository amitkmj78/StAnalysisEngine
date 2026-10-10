"""STS-3: a published strategy's ongoing forward paper track record, from
its publish date. This is a replay of the exact same backtest math the
builder itself uses (services/strategy_engine.py::run_backtest, unmodified)
over the real [published_at, today] window -- disclosed as a replay, not a
brokered paper account. Compare services/challenge_service.py, whose equity
snapshots come from a real linked Alpaca account; this does not.

Ticker-universe resolution mirrors web/backend/routers/strategy_builder.py::
backtest's own hand_picked/random_sample/portfolio branches exactly, so a
strategy's forward record always evaluates the same universe its backtest
did -- not a second, drifting implementation.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.notification_dispatcher import dispatch_alert
from services.stock_finder_service import fetch_sp500_tickers
from services.strategy_engine import feature_frame, run_backtest
from services.yfinance_cache import get_cached_history_range, get_earnings_report_dates
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

BENCHMARK = "SPY"
# Fewer than this many calendar days since publish means too few trading
# days for a real reading -- skipped (no row), never guessed at, same
# "too little data -> None" discipline as challenge_service.py::
# compute_member_performance.
MIN_DAYS_SINCE_PUBLISH = 2


def _resolve_tickers(definition: dict, universe: list[str]) -> list[str]:
    """Mirrors strategy_builder.py::backtest's own selection branches."""
    weights = definition.get("weights")
    if weights:
        return sorted({k.strip().upper() for k in weights})
    source = definition.get("source", "hand_picked")
    if source == "random_sample":
        size = definition.get("sample_size") or max(len(definition.get("tickers") or []), 5)
        seed = definition.get("sample_seed")
        if seed is None:
            return []  # no seed recorded -- can't reproduce the draw, fail open to "nothing to evaluate"
        tickers = random.Random(seed).sample(universe, min(size, len(universe)))
        return sorted(set(tickers))
    return sorted({t.strip().upper() for t in (definition.get("tickers") or []) if t.strip()})


async def _build_frames(tickers: list[str], start: str, end: str, regime_by_date: dict | None) -> dict:
    frames = {}
    for t in tickers:
        history = await run_in_threadpool(get_cached_history_range, t, start, end, True)
        if history.empty or not {"Open", "Close"}.issubset(history.columns):
            continue  # one missing ticker shouldn't block the rest -- same fail-open as every other per-ticker fetch in this app
        columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in history.columns]
        reports = await run_in_threadpool(get_earnings_report_dates, t)
        frames[t] = feature_frame(history[columns], regime_by_date, reports)
    return frames


async def _evaluate_one(row, universe: list[str], today: date) -> None:
    published_id = row["published_strategy_id"]
    published_at = row["published_at"].date()
    if (today - published_at).days < MIN_DAYS_SINCE_PUBLISH:
        return

    definition = row["definition"]
    entry_raw = definition.get("entry") or []
    exit_raw = definition.get("exit") or []
    if not entry_raw:
        return  # nothing to evaluate -- shouldn't happen for a real published strategy, but never guess a backtest

    tickers = _resolve_tickers(definition, universe)
    if not tickers:
        return

    uses_regime = any(r.get("field") == "regime" for r in entry_raw + exit_raw)
    regime_by_date = None
    if uses_regime:
        async with service_conn() as conn:
            regime_rows = await conn.fetch("SELECT as_of_date, regime_confirmed FROM market_regime_daily")
        regime_by_date = {r["as_of_date"].isoformat(): r["regime_confirmed"] for r in regime_rows}

    start = published_at.isoformat()
    end = today.isoformat()
    frames = await _build_frames(tickers, start, end, regime_by_date)
    if not frames:
        return
    bench_hist = await run_in_threadpool(get_cached_history_range, BENCHMARK, start, end, True)
    if bench_hist.empty:
        return

    weights = definition.get("weights")
    result = run_backtest(
        frames, entry_raw, exit_raw, bench_hist["Close"],
        exits_raw=definition.get("exits"),
        waive_protective_exit=bool(definition.get("waive_protective_exit")),
        cooldown=definition.get("cooldown_sessions", 0),
        verdict_benchmark=definition.get("verdict_benchmark", "basket"),
        weights={k.strip().upper(): float(v) for k, v in weights.items()} if weights else None,
    )
    cumulative_return_pct = (result.get("strategy") or {}).get("total_return_pct")
    if cumulative_return_pct is None:
        return
    trades = result.get("trades") or 0
    trade_log = result.get("trade_log") or []

    async with service_conn() as conn:
        prior_trades = await conn.fetchval(
            """
            SELECT trades FROM published_strategy_forward_snapshots
            WHERE published_strategy_id = $1 ORDER BY as_of_date DESC LIMIT 1
            """,
            published_id,
        )
        await conn.execute(
            """
            INSERT INTO published_strategy_forward_snapshots
                (published_strategy_id, as_of_date, cumulative_return_pct, trades)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (published_strategy_id, as_of_date) DO NOTHING
            """,
            published_id, today, cumulative_return_pct, trades,
        )

    # STS-5 (alerts half only -- never auto-paper-follow): a strategy's
    # followers get notified when THIS run detects newly-closed trades since
    # the last run. prior_trades is None on this strategy's first-ever
    # snapshot -- never alert then, since there is no real "new" baseline to
    # diff against, only the whole backtest history landing at once.
    new_trades = _new_trades_since(trade_log, prior_trades, trades)
    if new_trades:
        await _alert_followers_of_new_trades(published_id, row.get("name"), new_trades)


def _new_trades_since(trade_log: list[dict], prior_trades: int | None, trades: int) -> list[dict]:
    """trade_log is deterministic and chronological (see run_backtest), so
    newly-closed trades are always at the tail. prior_trades is None on a
    strategy's first-ever snapshot -- never alert then, since there is no
    real "new" baseline to diff against, only the whole backtest history
    landing at once."""
    if prior_trades is None or trades <= prior_trades:
        return []
    return trade_log[-(trades - prior_trades):]


async def _alert_followers_of_new_trades(published_id: int, name, new_trades: list[dict]) -> None:
    async with service_conn() as conn:
        followers = await conn.fetch(
            "SELECT user_id FROM strategy_follows WHERE published_strategy_id = $1", published_id
        )
    if not followers:
        return
    label = name or "A strategy you follow"
    for trade in new_trades:
        ticker = trade.get("ticker")
        return_pct = trade.get("return_pct")
        exit_reason = trade.get("exit_reason")
        summary = f"{ticker}: closed {return_pct}% ({exit_reason})" if return_pct is not None else f"{ticker}: new signal"
        async with service_conn() as conn:
            for f in followers:
                await conn.execute(
                    """
                    INSERT INTO strategy_signal_alerts (user_id, published_strategy_id, ticker, trade_summary)
                    VALUES ($1::uuid, $2, $3, $4)
                    """,
                    f["user_id"], published_id, ticker, summary,
                )
        for f in followers:
            await dispatch_alert(
                str(f["user_id"]), ticker, "strategy_signal",
                f"{label}: new signal on {ticker}",
                f"{label} just closed a trade on {ticker}: {summary}.",
                {"return_pct": return_pct, "exit_reason": exit_reason},
            )


async def evaluate_due_forward_records() -> int:
    """Scheduler entry point. One row attempted per published strategy per
    day; a failure on one strategy (bad ticker, no price history, a
    malformed historic definition) is logged and skipped, never aborting
    the rest -- same per-item isolation as portfolio_health_service.py's
    fan-outs."""
    today = date.today()
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT id AS published_strategy_id, published_at, definition, name
            FROM published_strategies
            WHERE NOT EXISTS (
                SELECT 1 FROM published_strategy_forward_snapshots s
                WHERE s.published_strategy_id = published_strategies.id AND s.as_of_date = $1
            )
            """,
            today,
        )
    if not rows:
        return 0

    rows = [
        {**dict(r), "definition": json.loads(r["definition"]) if isinstance(r["definition"], str) else r["definition"]}
        for r in rows
    ]

    universe: list[str] = []
    if any(_needs_sp500(r["definition"]) for r in rows):
        try:
            universe = sorted(await run_in_threadpool(fetch_sp500_tickers))
        except Exception as e:  # noqa: BLE001 -- the batch continues for strategies that don't need this
            logger.warning("Could not fetch S&P 500 membership for forward-record evaluation: %s", e)

    evaluated = 0
    for row in rows:
        try:
            await _evaluate_one(row, universe, today)
            evaluated += 1
        except Exception as e:  # noqa: BLE001 -- one bad strategy must never abort the batch
            logger.warning("Forward-record evaluation failed for published strategy %s: %s", row["published_strategy_id"], e)
    return evaluated


def _needs_sp500(definition: dict) -> bool:
    return not definition.get("weights") and definition.get("source") == "random_sample"
