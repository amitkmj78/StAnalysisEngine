"""
TR-7: the backtest engine's event-driven simulation loop. These tests
target the core execution logic in isolation from the yfinance fetch —
_run_event_driven_simulation takes a plain price matrix and is fully
deterministic given synthetic data, so bugs in the simulation's timing
(the actual hard part of "event-driven") are caught here without any
network dependency.
"""

import pandas as pd

from services.backtest_engine import run_event_driven_simulation as _run_event_driven_simulation


def _price_matrix(data: dict, days: int) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=days, freq="B")
    return pd.DataFrame(data, index=index)


def test_rebalance_day_itself_earns_no_mark_to_market_return():
    """
    Regression test for a same-day-lookahead bug caught during review: a
    newly-selected pick's price move on the rebalance day itself (i.e.
    before it was selected) must never be credited to that pick's return
    — the position is entered AT that day's close, so the earliest day it
    can actually earn a return is the next one.
    """
    days = 6
    a = [100, 100, 100, 200, 200, 200]  # jumps to 200 exactly on the rebalance day (index 3)
    b = [100, 100, 100, 100, 100, 100]  # flat — never selected
    price_matrix = _price_matrix({"A": a, "B": b}, days)

    periods, daily_strategy, daily_benchmark = _run_event_driven_simulation(
        price_matrix, lookback_days=3, top_n=1, horizon_days=3, slippage_bps=0.0, commission_bps=0.0,
    )

    assert len(periods) == 1
    assert periods[0]["picks"] == ["A"]
    # If the rebalance day's own 100% jump were wrongly credited, gross
    # return here would be ~100%. It must be 0%: A does nothing from the
    # day it was entered (day index 3) through the end of the period.
    assert periods[0]["strategy_return_gross_pct"] == 0.0


def test_picks_change_when_momentum_flips():
    days = 9
    a = [100, 100, 100, 110, 111, 112, 100, 90, 80]
    b = [100, 100, 100, 100, 101, 102, 130, 140, 150]
    price_matrix = _price_matrix({"A": a, "B": b}, days)

    periods, _, _ = _run_event_driven_simulation(
        price_matrix, lookback_days=3, top_n=1, horizon_days=3, slippage_bps=0.0, commission_bps=0.0,
    )

    assert [p["picks"] for p in periods] == [["A"], ["B"]]
    assert periods[0]["turnover_pct"] == 100.0  # initial purchase
    assert periods[1]["turnover_pct"] == 100.0  # full swap, A dropped for B


def test_costs_reduce_net_return_below_gross():
    days = 9
    a = [100, 100, 100, 110, 111, 112, 113, 114, 115]
    b = [100, 100, 100, 90, 90, 90, 90, 90, 90]
    price_matrix = _price_matrix({"A": a, "B": b}, days)

    periods, _, _ = _run_event_driven_simulation(
        price_matrix, lookback_days=3, top_n=1, horizon_days=3, slippage_bps=50.0, commission_bps=10.0,
    )

    assert len(periods) >= 1
    p = periods[0]
    assert p["strategy_return_pct"] < p["strategy_return_gross_pct"]


def test_no_data_produces_no_periods_not_a_crash():
    price_matrix = _price_matrix({"A": [], "B": []}, 0)
    periods, daily_strategy, daily_benchmark = _run_event_driven_simulation(
        price_matrix, lookback_days=3, top_n=1, horizon_days=3, slippage_bps=0.0, commission_bps=0.0,
    )
    assert periods == []
    assert daily_strategy == []
    assert daily_benchmark == []
