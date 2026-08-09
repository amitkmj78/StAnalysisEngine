"""
TR-7: the event-driven backtest simulation core, deliberately kept free
of any import that pulls in the app's heavier service chains (stock/fund
universe resolution transitively initializes LLM clients at import time)
— this module only needs stdlib + numpy/pandas, so it's fast and
dependency-free to unit test in CI. momentum_backtest_service.py is the
caller that supplies actual price data and universe tickers.
"""

from typing import Optional

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252  # for annualizing daily-return risk metrics


def cumulative_pct(returns_pct: list[float]) -> Optional[float]:
    if not returns_pct:
        return None
    total = 1.0
    for r in returns_pct:
        total *= 1 + r / 100
    return round((total - 1) * 100, 2)


def max_drawdown_pct(returns_pct: list[float]) -> Optional[float]:
    if not returns_pct:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns_pct:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    return round(max_dd * 100, 2)


def sharpe(returns_pct: list[float], risk_free_rate_annual: float, periods_per_year: float) -> Optional[float]:
    if len(returns_pct) < 2:
        return None
    returns = np.array(returns_pct) / 100
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return None
    return round(float(np.mean(excess)) / std * np.sqrt(periods_per_year), 2)


def sortino(returns_pct: list[float], risk_free_rate_annual: float, periods_per_year: float) -> Optional[float]:
    if len(returns_pct) < 2:
        return None
    returns = np.array(returns_pct) / 100
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None  # never had a losing day — Sortino is undefined, not infinite
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(np.abs(downside[0]))
    if downside_std == 0:
        return None
    return round(float(np.mean(excess)) / downside_std * np.sqrt(periods_per_year), 2)


def run_event_driven_simulation(
    price_matrix: pd.DataFrame,
    lookback_days: int,
    top_n: int,
    horizon_days: int,
    slippage_bps: float,
    commission_bps: float,
) -> tuple[list[dict], list[float], list[float]]:
    """
    TR-7: event-driven, not vectorized — steps through trading days one at
    a time maintaining explicit portfolio state (current picks, in-progress
    period P&L), rather than computing returns as a single bulk operation
    over precomputed rebalance points. At every day, only price rows up to
    and including that day are ever read — the loop's own structure
    enforces no-lookahead, the same guarantee the PIT reconciliation
    mechanism relies on elsewhere in this app, just applied here to
    execution instead of ranking.

    Returns (periods, daily_strategy_returns_pct, daily_benchmark_returns_pct).
    Only complete periods (a full horizon_days held) are reported — a
    trailing partial period at the end of the simulated range is dropped
    rather than reported as if it were comparable to a full one.
    """
    round_trip_cost_pct = 2 * (slippage_bps + commission_bps) / 100  # bps -> %, doubled for entry+exit

    periods: list[dict] = []
    daily_strategy_returns: list[float] = []
    daily_benchmark_returns: list[float] = []

    current_picks: list[str] = []
    prev_picks_set: Optional[set[str]] = None
    turnover_frac = 1.0

    period_start_idx: Optional[int] = None
    period_gross_factor = 1.0
    period_benchmark_factor = 1.0
    period_cost_pct = 0.0
    period_day_count = 0

    def close_period() -> None:
        # A full period accrues horizon_days - 1 mark-to-market days: the
        # rebalance day itself enters the new book at that day's close but
        # earns no P&L that day (see the loop below), so the first return
        # day is the one after it.
        if period_day_count < horizon_days - 1:
            return  # incomplete trailing period — not comparable to a full one, drop it
        strategy_return_gross = round((period_gross_factor - 1) * 100, 2)
        strategy_return = round(strategy_return_gross - period_cost_pct, 2)
        benchmark_return = round((period_benchmark_factor - 1) * 100, 2)
        periods.append({
            "date": price_matrix.index[period_start_idx].strftime("%Y-%m-%d"),
            "picks": current_picks,
            "strategy_return_pct": strategy_return,
            "strategy_return_gross_pct": strategy_return_gross,
            "benchmark_return_pct": benchmark_return,
            "turnover_pct": round(turnover_frac * 100, 1),
        })

    n = len(price_matrix)
    day = lookback_days
    while day < n:
        is_rebalance_day = (day - lookback_days) % horizon_days == 0

        if is_rebalance_day:
            if period_start_idx is not None:
                close_period()

            # EVENT: rank using only data known up to and including `day`,
            # then enter the new book AT this day's close. The position
            # isn't "live" for mark-to-market purposes until the next day
            # — crediting it with the price move that happened before it
            # was even selected would be a same-day lookahead bug.
            lookback_start = price_matrix.iloc[day - lookback_days]
            lookback_now = price_matrix.iloc[day]
            momentum = (lookback_now / lookback_start - 1.0).dropna()
            if len(momentum) < top_n:
                break  # can't form a valid book from here on — stop, don't fabricate

            new_picks = momentum.sort_values(ascending=False).head(top_n).index.tolist()
            new_picks_set = set(new_picks)
            turnover_frac = 1.0 if prev_picks_set is None else len(new_picks_set - prev_picks_set) / len(new_picks_set)
            prev_picks_set = new_picks_set
            current_picks = new_picks

            period_start_idx = day
            period_gross_factor = 1.0
            period_benchmark_factor = 1.0
            period_cost_pct = turnover_frac * round_trip_cost_pct
            period_day_count = 0
        elif current_picks:
            # EVENT: end-of-day mark-to-market, strictly yesterday -> today,
            # for a book that's already been held since a prior close.
            prev_row = price_matrix.iloc[day - 1]
            today_row = price_matrix.iloc[day]

            pick_returns = (today_row[current_picks] / prev_row[current_picks] - 1.0).dropna()
            if not pick_returns.empty:
                daily_ret = float(pick_returns.mean())
                daily_strategy_returns.append(round(daily_ret * 100, 4))
                period_gross_factor *= (1 + daily_ret)
                period_day_count += 1

            universe_returns = (today_row / prev_row - 1.0).dropna()
            if not universe_returns.empty:
                daily_bmk = float(universe_returns.mean())
                daily_benchmark_returns.append(round(daily_bmk * 100, 4))
                period_benchmark_factor *= (1 + daily_bmk)

        day += 1

    if period_start_idx is not None:
        close_period()

    return periods, daily_strategy_returns, daily_benchmark_returns
