"""
Portfolio vs. Top Picks compare page (new /portfolio/compare endpoint).
No live network -- every test builds synthetic price series and
monkeypatches the module's data-fetching functions, matching this
repo's established convention (see tests/test_stock_finder_service.py,
tests/test_diversified_basket.py).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

import services.portfolio_compare_service as pcs
from services.portfolio_compare_service import (
    HoldingInput,
    WindowBounds,
    _rebased_series,
    build_headline,
    build_portfolio_window_view,
    derive_confidence,
    mark_owned,
    resolve_window,
    select_gap_drivers,
)


def _prices(values, start="2024-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


def _hist_df(prices: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Close": prices})


# ---------------------------------------------------------------------------
# resolve_window
# ---------------------------------------------------------------------------


def test_resolve_window_trading_day_counts(monkeypatch):
    ref = _prices([100.0 + i for i in range(500)])
    monkeypatch.setattr(pcs, "get_cached_history", lambda ticker, period, auto_adjust=True: _hist_df(ref))

    bounds = resolve_window("30D")
    assert bounds.trading_days == 30
    assert bounds.end == ref.index[-1]
    assert bounds.start == ref.index[-31]


def test_resolve_window_unknown_code_raises():
    with pytest.raises(ValueError):
        resolve_window("7D")


def test_resolve_window_insufficient_history_raises(monkeypatch):
    ref = _prices([100.0, 101.0, 102.0])
    monkeypatch.setattr(pcs, "get_cached_history", lambda ticker, period, auto_adjust=True: _hist_df(ref))
    with pytest.raises(ValueError):
        resolve_window("90D")


# ---------------------------------------------------------------------------
# _rebased_series
# ---------------------------------------------------------------------------


def test_rebased_series_starts_at_10000():
    prices = _prices([137.5, 140.0, 150.0])
    series = _rebased_series(prices)
    assert series[0][1] == pytest.approx(10000.0)
    assert series[-1][1] == pytest.approx(10000.0 * (150.0 / 137.5))


def test_rebased_series_empty_is_a_noop():
    assert _rebased_series(pd.Series(dtype=float)) == []


# ---------------------------------------------------------------------------
# build_portfolio_window_view
# ---------------------------------------------------------------------------


def test_contribution_sums_to_portfolio_return(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    aaa = pd.Series(np.linspace(100, 110, 40), index=idx)  # +10%
    bbb = pd.Series(np.linspace(50, 45, 40), index=idx)  # -10%

    def fake_hist(ticker, period, auto_adjust=True):
        return _hist_df({"AAA": aaa, "BBB": bbb}[ticker])

    monkeypatch.setattr(pcs, "get_cached_history", fake_hist)
    bounds = WindowBounds("30D", idx[0], idx[-1], 30)
    holdings = [HoldingInput("AAA", 10, 100.0, None), HoldingInput("BBB", 20, 50.0, None)]

    view = build_portfolio_window_view(holdings, bounds)
    total_contribution = sum(h["contribution_pts"] for h in view["holdings"])
    assert total_contribution == pytest.approx(view["return_pct"], abs=0.0001)


def test_portfolio_series_starts_at_10000(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    aaa = pd.Series(np.linspace(100, 120, 40), index=idx)
    monkeypatch.setattr(pcs, "get_cached_history", lambda ticker, period, auto_adjust=True: _hist_df(aaa))
    bounds = WindowBounds("30D", idx[0], idx[-1], 30)
    view = build_portfolio_window_view([HoldingInput("AAA", 10, 100.0, None)], bounds)
    assert view["series"][0][1] == pytest.approx(10000.0)


def test_cash_counts_as_zero_return_holding(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    aaa = pd.Series(np.linspace(100, 110, 40), index=idx)
    monkeypatch.setattr(pcs, "get_cached_history", lambda ticker, period, auto_adjust=True: _hist_df(aaa))
    bounds = WindowBounds("30D", idx[0], idx[-1], 30)
    view = build_portfolio_window_view([HoldingInput("AAA", 10, 100.0, None)], bounds, cash_balance=1000.0)

    cash_row = next(h for h in view["holdings"] if h["ticker"] == "CASH")
    assert cash_row["return_pct"] == 0.0
    assert cash_row["contribution_pts"] == 0.0
    assert cash_row["weight_pct"] == pytest.approx(50.0)  # 1000 cash / (1000 AAA + 1000 cash)


def test_holding_bought_inside_window_gets_since_date(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    # Deliberately different trajectories before/after day 10 so the
    # window-start-anchored vs. purchase-date-anchored return differ.
    prices = pd.Series([100.0] * 10 + list(np.linspace(100, 200, 20)), index=idx)
    monkeypatch.setattr(pcs, "get_cached_history", lambda ticker, period, auto_adjust=True: _hist_df(prices))
    bounds = WindowBounds("30D", idx[0], idx[-1], 30)
    acquired = idx[10].date()
    holdings = [HoldingInput("AAA", 5, 150.0, acquired)]

    view = build_portfolio_window_view(holdings, bounds)
    row = view["holdings"][0]
    assert row["since"] == acquired.isoformat()
    # From day 10 (100.0) to the end (200.0) is +100%, not the ~0%
    # window-start-anchored figure a naive full-window slice would give.
    assert row["return_pct"] == pytest.approx(100.0, abs=1.0)


def test_excluded_ticker_with_no_history_does_not_crash(monkeypatch):
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    aaa = pd.Series(np.linspace(100, 110, 30), index=idx)

    def fake_hist(ticker, period, auto_adjust=True):
        return _hist_df(aaa) if ticker == "AAA" else pd.DataFrame()

    monkeypatch.setattr(pcs, "get_cached_history", fake_hist)
    bounds = WindowBounds("30D", idx[0], idx[-1], 30)
    holdings = [HoldingInput("AAA", 10, 100.0, None), HoldingInput("DELISTED", 5, 20.0, None)]

    view = build_portfolio_window_view(holdings, bounds)
    assert "DELISTED" in view["excluded_from_risk"]
    delisted_row = next(h for h in view["holdings"] if h["ticker"] == "DELISTED")
    assert delisted_row["return_pct"] is None
    assert delisted_row["contribution_pts"] == 0.0


def test_empty_holdings_returns_none_not_crash():
    bounds = WindowBounds("30D", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), 30)
    view = build_portfolio_window_view([], bounds)
    assert view["return_pct"] is None
    assert view["holdings"] == []
    assert view["series"] == []


# ---------------------------------------------------------------------------
# select_gap_drivers
# ---------------------------------------------------------------------------


def test_gap_drivers_selects_lead_and_two_drags():
    holdings = [
        {"ticker": "A", "contribution_pts": 3.0},
        {"ticker": "B", "contribution_pts": -0.5},
        {"ticker": "C", "contribution_pts": -2.0},
        {"ticker": "D", "contribution_pts": -1.0},
        {"ticker": "E", "contribution_pts": 0.1},
    ]
    drivers = select_gap_drivers(holdings)
    assert drivers == [
        {"ticker": "A", "kind": "lead", "contribution_pts": 3.0},
        {"ticker": "C", "kind": "drag", "contribution_pts": -2.0},
        {"ticker": "D", "kind": "drag", "contribution_pts": -1.0},
    ]


def test_gap_drivers_fewer_than_two_negative():
    holdings = [{"ticker": "A", "contribution_pts": 3.0}, {"ticker": "B", "contribution_pts": -0.5}]
    drivers = select_gap_drivers(holdings)
    assert len(drivers) == 2
    assert drivers[1] == {"ticker": "B", "kind": "drag", "contribution_pts": -0.5}


def test_gap_drivers_deterministic_tiebreak():
    holdings = [{"ticker": "ZZZ", "contribution_pts": 1.0}, {"ticker": "AAA", "contribution_pts": 1.0}]
    drivers = select_gap_drivers(holdings)
    assert drivers[0]["ticker"] == "AAA"  # alphabetical tie-break


def test_gap_drivers_no_positive_no_negative():
    holdings = [{"ticker": "A", "contribution_pts": 0.0}]
    assert select_gap_drivers(holdings) == []


# ---------------------------------------------------------------------------
# build_headline
# ---------------------------------------------------------------------------


def test_headline_all_present():
    text = build_headline("90 days", 4.0, 5.1, "VTI", 5.3)
    assert text == "Your portfolio is up 4.0% over 90 days — 1.1 pts behind the S&P 500 and 1.3 pts behind VTI."


def test_headline_none_portfolio_return():
    assert build_headline("90 days", None, 5.1, "VTI", 5.3) == "Not enough price history yet to summarize this window."


def test_headline_missing_benchmark_omits_clause():
    text = build_headline("90 days", 4.0, None, "VTI", 5.3)
    assert "S&P 500" not in text
    assert "VTI" in text


def test_headline_missing_top_fund_omits_clause():
    text = build_headline("90 days", 4.0, 5.1, None, None)
    assert "S&P 500" in text
    assert "pick" not in text  # no fund clause at all


def test_headline_negative_return_uses_down():
    text = build_headline("90 days", -2.5, None, None, None)
    assert text.startswith("Your portfolio is down 2.5%")


# ---------------------------------------------------------------------------
# derive_confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stability,expected_label,expected_score",
    [
        (None, "unknown", None),
        ({"flip_count": 0, "unstable": False}, "high", 100),
        ({"flip_count": 1, "unstable": False}, "high", 75),
        ({"flip_count": 2, "unstable": False}, "medium", 50),
        ({"flip_count": 3, "unstable": True}, "low", 25),
        ({"flip_count": 10, "unstable": True}, "low", 0),
    ],
)
def test_derive_confidence_mapping(stability, expected_label, expected_score):
    result = derive_confidence(stability)
    assert result["label"] == expected_label
    assert result["score"] == expected_score


def test_derive_confidence_label_score_never_disagree():
    for flip_count in range(10):
        for unstable in (True, False):
            result = derive_confidence({"flip_count": flip_count, "unstable": unstable})
            score = result["score"]
            if score >= 75:
                assert result["label"] == "high"
            elif score >= 40:
                assert result["label"] == "medium"
            else:
                assert result["label"] == "low"


# ---------------------------------------------------------------------------
# mark_owned
# ---------------------------------------------------------------------------


def test_mark_owned():
    rows = [{"ticker": "AAPL"}, {"ticker": "PLTR"}]
    result = mark_owned(rows, {"AAPL"})
    assert result[0]["owned"] is True
    assert result[1]["owned"] is False
