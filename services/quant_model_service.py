"""Quant model as a challenge member: a hypothetical daily portfolio.

Each publication day the model publishes its ranked picks (published_signals).
The hypothetical portfolio holds that day's picks equal-weight until the next
publication, then rebalances. Returns come from closing prices. No orders,
fills, or slippage beyond the cost assumption are modelled, so this is a
paper-style track record, not a live one. It is always shown as hypothetical.

Costs use the same 10 bps one-way assumption as the TRK-6 model portfolio,
charged once per rebalance on the full cohort (20 bps round trip).
"""

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from starlette.concurrency import run_in_threadpool

from services.signal_publication_service import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_UNIVERSE,
    MODEL_PORTFOLIO_COST_BPS_ONE_WAY,
    MODEL_PORTFOLIO_REBASE_TO,
)
from services.yfinance_cache import get_cached_history
from web.backend.db import service_conn

MODEL_MEMBER_LABEL = "Quant model (hypothetical)"


def daily_model_returns(
    picks: dict[date, list[str]],
    closes: dict[str, pd.Series],
    cost_bps_one_way: float = MODEL_PORTFOLIO_COST_BPS_ONE_WAY,
) -> list[tuple[date, float]]:
    """Pure. Returns [(date, net_return_pct)] for each rebalance day after the
    first. A day's return is the mean price return of its picks, from that
    publication's close to the next publication's close, less the rebalance
    cost. Picks with a missing price on either date are dropped from the mean,
    and a day with no priced picks is skipped rather than guessed."""
    cost_fraction = 2 * cost_bps_one_way / 10_000
    dates = sorted(picks)
    out: list[tuple[date, float]] = []
    for d0, d1 in zip(dates, dates[1:]):
        returns = []
        for ticker in picks[d0]:
            series = closes.get(ticker)
            if series is None:
                continue
            if pd.Timestamp(d0) not in series.index or pd.Timestamp(d1) not in series.index:
                continue
            p0 = float(series.loc[pd.Timestamp(d0)])
            p1 = float(series.loc[pd.Timestamp(d1)])
            if p0 > 0:
                returns.append(p1 / p0 - 1.0)
        if not returns:
            continue
        mean = sum(returns) / len(returns)
        net = (1.0 + mean) * (1.0 - cost_fraction) - 1.0
        out.append((d1, round(net * 100.0, 4)))
    return out


def as_calendar_dates(series: pd.Series) -> pd.Series:
    """Price history is indexed at New York midnight (timezone-aware), while
    publication dates are plain dates. Lookups only match on plain calendar
    dates, so every series is normalised to that before use."""
    index = series.index
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    return pd.Series(series.values, index=index.normalize())


def equity_snapshots_from_returns(returns: list[tuple[date, float]]) -> list[dict]:
    """Turns daily % returns into the {as_of_date, equity} shape that
    compute_member_performance and rebase_to_100 already take, compounding
    from MODEL_PORTFOLIO_REBASE_TO. Callers cut the window afterwards so the
    baseline is the first in-window publication, as for human members."""
    if not returns:
        return []
    value = MODEL_PORTFOLIO_REBASE_TO
    snaps = []
    for d, pct in returns:
        value *= 1.0 + pct / 100.0
        snaps.append({"as_of_date": d, "equity": value})
    return snaps


async def fetch_picks(start: date, end: date) -> dict[date, list[str]]:
    """Latest published pick set per target date for the default universe and
    lookback. The day before `start` is included so the first in-window return
    has a holding to measure from."""
    lookback_start = start - timedelta(days=7)
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (target_date, ticker) target_date, ticker
            FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND target_date BETWEEN $3 AND $4
            ORDER BY target_date, ticker, published_at_utc DESC
            """,
            DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS, lookback_start, end,
        )
    picks: dict[date, list[str]] = {}
    for r in rows:
        picks.setdefault(r["target_date"], []).append(r["ticker"])
    return picks


async def model_snapshots(start: date, end: date) -> list[dict]:
    """Daily hypothetical equity for the challenge window, or [] if no
    publication history covers it."""
    picks = await fetch_picks(start, end)
    if len(picks) < 2:
        return []
    tickers = sorted({t for ts in picks.values() for t in ts})
    closes: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            frame = await run_in_threadpool(get_cached_history, t, "1y", True)
            closes[t] = as_calendar_dates(frame["Close"].dropna())
        except Exception:  # noqa: BLE001 -- a missing ticker drops out of the mean; never fail the board
            continue
    chain = equity_snapshots_from_returns(daily_model_returns(picks, closes))
    return [snap for snap in chain if start <= snap["as_of_date"] <= end]
