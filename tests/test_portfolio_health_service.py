"""
Portfolio Health Check (HLT-1..4). No live network -- every test builds
synthetic price series and monkeypatches the module's data-fetching
functions, matching this repo's established convention (see
tests/test_diversified_basket.py::test_compute_basket_risk_preview_*,
tests/test_portfolio_compare_service.py).
"""

import pandas as pd
import pytest

import services.portfolio_health_service as phs
from services.portfolio_health_service import (
    build_sector_comparison,
    compute_portfolio_risk_metrics,
    compute_portfolio_sector_weights,
    compute_risk_over_windows,
)


def _prices(values, start="2023-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# compute_portfolio_risk_metrics
# ---------------------------------------------------------------------------


def test_compute_portfolio_risk_metrics_beta_and_correlation_match_construction(monkeypatch):
    # SPY moves by `spy_moves` each day; the portfolio's one holding moves
    # at exactly 1.5x SPY's daily moves by construction -- beta must come
    # back ~1.5, and since it's a noiseless linear multiple, correlation
    # must come back ~1.0.
    spy_moves = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.005, 0.01] * 5
    spy_prices = [100.0]
    for m in spy_moves:
        spy_prices.append(spy_prices[-1] * (1 + m))
    stock_prices = [50.0]
    for m in spy_moves:
        stock_prices.append(stock_prices[-1] * (1 + 1.5 * m))

    def fake_history(ticker, period, auto_adjust=True):
        if ticker == "SPY":
            return pd.DataFrame({"Close": _prices(spy_prices)})
        return pd.DataFrame({"Close": _prices(stock_prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    result = compute_portfolio_risk_metrics([{"ticker": "LEVERED", "market_value": 10_000.0}], period="1y")
    assert result["beta_to_spy"] == pytest.approx(1.5, abs=0.05)
    assert result["correlation_to_spy"] == pytest.approx(1.0, abs=0.01)
    assert result["excluded_from_risk"] == []
    assert result["period"] == "1y"
    assert result["data_start"] is not None
    assert result["data_end"] is not None


def test_compute_portfolio_risk_metrics_identical_to_spy_gives_beta_and_correlation_one(monkeypatch):
    spy_moves = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.005, 0.01] * 5
    spy_prices = [100.0]
    for m in spy_moves:
        spy_prices.append(spy_prices[-1] * (1 + m))

    monkeypatch.setattr(
        phs, "get_cached_history",
        lambda ticker, period, auto_adjust=True: pd.DataFrame({"Close": _prices(spy_prices)}),
    )

    result = compute_portfolio_risk_metrics([{"ticker": "SPYCLONE", "market_value": 5_000.0}], period="1y")
    assert result["beta_to_spy"] == pytest.approx(1.0, abs=0.01)
    assert result["correlation_to_spy"] == pytest.approx(1.0, abs=0.01)


def test_compute_portfolio_risk_metrics_excludes_ticker_with_no_history(monkeypatch):
    spy_prices = [100.0 + i for i in range(30)]

    def fake_history(ticker, period, auto_adjust=True):
        if ticker == "NODATA":
            return pd.DataFrame()
        return pd.DataFrame({"Close": _prices(spy_prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    result = compute_portfolio_risk_metrics(
        [{"ticker": "GOOD", "market_value": 5_000.0}, {"ticker": "NODATA", "market_value": 5_000.0}],
        period="1y",
    )
    assert "NODATA" in result["excluded_from_risk"]
    assert result["beta_to_spy"] is not None


def test_compute_portfolio_risk_metrics_empty_positions():
    result = compute_portfolio_risk_metrics([], period="1y")
    assert result["beta_to_spy"] is None
    assert result["volatility_pct"] is None
    assert result["excluded_from_risk"] == []


def test_compute_portfolio_risk_metrics_missing_spy_data_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(phs, "get_cached_history", lambda ticker, period, auto_adjust=True: pd.DataFrame())
    result = compute_portfolio_risk_metrics([{"ticker": "A", "market_value": 1_000.0}], period="1y")
    assert result["beta_to_spy"] is None
    assert "A" in result["excluded_from_risk"]


def test_compute_risk_over_windows_calls_both_periods(monkeypatch):
    periods_requested = []

    def fake_history(ticker, period, auto_adjust=True):
        if ticker != "SPY":
            periods_requested.append(period)
        prices = [100.0 + i for i in range(30)]
        return pd.DataFrame({"Close": _prices(prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    result = compute_risk_over_windows([{"ticker": "A", "market_value": 1_000.0}])
    assert set(result.keys()) == {"1Y", "3Y"}
    assert result["1Y"]["period"] == "1y"
    assert result["3Y"]["period"] == "3y"
    assert "1y" in periods_requested
    assert "3y" in periods_requested


# ---------------------------------------------------------------------------
# compute_portfolio_sector_weights -- must apply the same GICS renaming
# compute_sp500_sector_mix uses, or the two sides of HLT-1's sector
# comparison silently fail to line up.
# ---------------------------------------------------------------------------


def test_compute_portfolio_sector_weights_applies_gics_renaming():
    positions = [
        {"ticker": "AAPL", "sector": "Technology", "market_value": 6_000.0},
        {"ticker": "JNJ", "sector": "Healthcare", "market_value": 4_000.0},
    ]
    result = compute_portfolio_sector_weights(positions)
    # Raw Yahoo names ("Technology", "Healthcare") must be renamed to
    # their GICS equivalents ("Information Technology", "Health Care") --
    # the same mapping stock_finder_service.compute_sp500_sector_mix uses.
    assert result == {"Information Technology": 60.0, "Health Care": 40.0}


def test_compute_portfolio_sector_weights_excludes_unsectored_from_numerator_only():
    positions = [
        {"ticker": "AAPL", "sector": "Technology", "market_value": 5_000.0},
        {"ticker": "SPY", "sector": None, "market_value": 5_000.0},
    ]
    result = compute_portfolio_sector_weights(positions)
    # SPY (no sector) is excluded from the breakdown but its value still
    # counts in the denominator -- AAPL's weight is 50%, not 100%.
    assert result == {"Information Technology": 50.0}


def test_compute_portfolio_sector_weights_empty_positions():
    assert compute_portfolio_sector_weights([]) == {}


# ---------------------------------------------------------------------------
# build_sector_comparison
# ---------------------------------------------------------------------------


def test_build_sector_comparison_union_and_gap():
    portfolio_weights = {"Information Technology": 60.0, "Health Care": 10.0}
    sp500_weights = {"Information Technology": 30.0, "Financials": 15.0}
    result = build_sector_comparison(portfolio_weights, sp500_weights)
    by_sector = {r["sector"]: r for r in result}

    assert by_sector["Information Technology"]["gap_pct"] == pytest.approx(30.0)
    assert by_sector["Health Care"]["sp500_weight_pct"] == 0.0
    assert by_sector["Financials"]["portfolio_weight_pct"] == 0.0
    assert by_sector["Financials"]["gap_pct"] == pytest.approx(-15.0)
    assert set(by_sector.keys()) == {"Information Technology", "Health Care", "Financials"}


def test_build_sector_comparison_empty_both_sides():
    assert build_sector_comparison({}, {}) == []
