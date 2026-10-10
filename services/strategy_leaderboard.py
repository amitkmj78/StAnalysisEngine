"""STS-4: a leaderboard of published strategies ranked by forward
risk-adjusted excess return vs SPY, gated on a minimum of 3 months and 30
trades. Reads services/strategy_forward_record.py's own daily snapshots
(published_strategy_forward_snapshots) -- no new storage, no new backtest.

"Risk-adjusted excess return vs SPY" is the information ratio: each
strategy's own day-over-day returns (derived from its cumulative_return_pct
series) minus SPY's day-over-day returns over the identical window, then
services/backtest_engine.py::sharpe() applied to that excess series with
risk_free_rate_annual=0.0 (the excess is already relative to the benchmark,
so there is nothing further to subtract) -- the same sharpe() helper
services/challenge_service.py::compute_member_performance already calls the
same way, just fed an excess series instead of a raw one.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from starlette.concurrency import run_in_threadpool

from services.backtest_engine import DAYS_PER_YEAR, cumulative_pct, sharpe
from services.yfinance_cache import get_cached_history
from web.backend.db import service_conn

MIN_LEADERBOARD_DAYS = 90
MIN_LEADERBOARD_TRADES = 30
BENCHMARK = "SPY"
NOT_ENOUGH_DATA = "not enough data yet"


def _daily_returns_from_cumulative(cumulative_series: list[float]) -> list[float]:
    """cumulative_series[i] is the total return as of day i, baselined at 0%
    on the (implicit) day before the first snapshot -- so day 0's own return
    is just cumulative_series[0], and day i>0's is the day-over-day change in
    (1 + cumulative/100)."""
    if not cumulative_series:
        return []
    equity = [1.0 + c / 100.0 for c in cumulative_series]
    out = [cumulative_series[0]]
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        out.append((equity[i] / prev - 1.0) * 100.0 if prev else 0.0)
    return out


async def _spy_daily_returns_for_dates(as_of_dates: list[date]) -> list[float]:
    """SPY's own day-over-day return for each as_of_date, baselined the same
    way the strategy's own series is (first date = cumulative return since
    the window start). Series.asof falls back to the latest prior trading
    day, so a non-trading as_of_date never raises. Empty if price data is
    missing, so the caller falls back to "not enough data yet" rather than
    guessing a benchmark."""
    closes = (await run_in_threadpool(get_cached_history, BENCHMARK, "2y", True))["Close"]
    closes.index = pd.to_datetime(closes.index)
    values = [closes.asof(pd.Timestamp(d)) for d in as_of_dates]
    if values[0] is None or pd.isna(values[0]) or not float(values[0]):
        return []
    base = float(values[0])
    cumulative_series = []
    for v in values:
        if v is None or pd.isna(v):
            return []
        cumulative_series.append((float(v) / base - 1.0) * 100.0)
    return _daily_returns_from_cumulative(cumulative_series)


async def build_strategy_leaderboard() -> list[dict]:
    today = date.today()
    async with service_conn() as conn:
        strategies = await conn.fetch(
            """
            SELECT * FROM (
                SELECT DISTINCT ON (COALESCE(p.root_published_id, p.id))
                       p.id, p.name, p.published_at, u.display_name
                FROM published_strategies p
                LEFT JOIN users u ON u.id = p.author_user_id
                ORDER BY COALESCE(p.root_published_id, p.id), p.version DESC
            ) latest
            """
        )
        snapshot_rows = {
            s["id"]: await conn.fetch(
                """
                SELECT as_of_date, cumulative_return_pct, trades
                FROM published_strategy_forward_snapshots
                WHERE published_strategy_id = $1 ORDER BY as_of_date
                """,
                s["id"],
            )
            for s in strategies
        }

    entries: list[dict] = []
    for s in strategies:
        published_at = s["published_at"].date() if isinstance(s["published_at"], datetime) else s["published_at"]
        snapshots = snapshot_rows[s["id"]]
        base_entry = {
            "id": s["id"], "name": s["name"], "author_display_name": s["display_name"],
            "published_at": published_at.isoformat(),
        }
        days_since_publish = (today - published_at).days
        latest_trades = snapshots[-1]["trades"] if snapshots else 0
        latest_return = snapshots[-1]["cumulative_return_pct"] if snapshots else None
        eligible = days_since_publish >= MIN_LEADERBOARD_DAYS and latest_trades >= MIN_LEADERBOARD_TRADES

        if eligible and snapshots:
            as_of_dates = [r["as_of_date"] for r in snapshots]
            strategy_daily = _daily_returns_from_cumulative([r["cumulative_return_pct"] for r in snapshots])
            spy_daily = await _spy_daily_returns_for_dates(as_of_dates)
            if spy_daily and len(spy_daily) == len(strategy_daily):
                excess_daily = [sd - bd for sd, bd in zip(strategy_daily, spy_daily)]
                spy_return_pct_total = cumulative_pct(spy_daily)
                entries.append({
                    **base_entry,
                    "eligible": True,
                    "reason": None,
                    "risk_adjusted_excess_return": sharpe(excess_daily, 0.0, DAYS_PER_YEAR),
                    "cumulative_return_pct": latest_return,
                    "excess_return_pct": (
                        round(latest_return - spy_return_pct_total, 2) if spy_return_pct_total is not None else None
                    ),
                    "trades": latest_trades,
                })
                continue

        entries.append({
            **base_entry, "eligible": False, "reason": NOT_ENOUGH_DATA,
            "risk_adjusted_excess_return": None, "cumulative_return_pct": latest_return,
            "excess_return_pct": None, "trades": latest_trades,
        })

    eligible_entries = sorted(
        (e for e in entries if e["eligible"]),
        key=lambda e: e["risk_adjusted_excess_return"] if e["risk_adjusted_excess_return"] is not None else float("-inf"),
        reverse=True,
    )
    ineligible_entries = [e for e in entries if not e["eligible"]]
    return eligible_entries + ineligible_entries
