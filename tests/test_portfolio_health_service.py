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
    compute_fund_coverage_pct,
    compute_look_through_exposure,
    compute_portfolio_risk_metrics,
    compute_portfolio_sector_weights,
    compute_risk_over_windows,
    fetch_fund_holdings_map,
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


def test_compute_portfolio_risk_metrics_excludes_ticker_whose_fetch_raises(monkeypatch):
    # Regression: a real live case -- get_cached_history doesn't fail
    # open like its sibling yfinance_cache functions, and raised
    # YFRateLimitError straight through, 500-ing the whole endpoint over
    # one rate-limited ticker in the fan-out.
    spy_prices = [100.0 + i for i in range(30)]

    def fake_history(ticker, period, auto_adjust=True):
        if ticker == "RATELIMITED":
            raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")
        return pd.DataFrame({"Close": _prices(spy_prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    result = compute_portfolio_risk_metrics(
        [{"ticker": "GOOD", "market_value": 5_000.0}, {"ticker": "RATELIMITED", "market_value": 5_000.0}],
        period="1y",
    )
    assert "RATELIMITED" in result["excluded_from_risk"]
    assert result["beta_to_spy"] is not None


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


# ---------------------------------------------------------------------------
# fetch_fund_holdings_map / compute_fund_coverage_pct / compute_look_through_exposure
# ---------------------------------------------------------------------------


def _fake_top_holdings(ticker_to_holdings: dict[str, dict[str, float]]):
    def fake(ticker: str) -> pd.DataFrame:
        holdings = ticker_to_holdings.get(ticker)
        if not holdings:
            return pd.DataFrame()
        return pd.DataFrame({"Holding Percent": holdings})

    return fake


def test_fetch_fund_holdings_map_only_keys_funds(monkeypatch):
    monkeypatch.setattr(
        phs, "get_cached_fund_top_holdings",
        _fake_top_holdings({"SPY": {"NVDA": 0.08, "AAPL": 0.07}}),
    )
    result = fetch_fund_holdings_map(["SPY", "AAPL"])
    assert set(result.keys()) == {"SPY"}
    assert result["SPY"] == {"NVDA": 0.08, "AAPL": 0.07}


def test_fetch_fund_holdings_map_empty_input():
    assert fetch_fund_holdings_map([]) == {}


def test_compute_fund_coverage_pct_sums_fractions_to_a_percent():
    fund_holdings = {"SPY": {"NVDA": 0.08, "AAPL": 0.07, "MSFT": 0.05}}
    result = compute_fund_coverage_pct(fund_holdings)
    assert result == {"SPY": 20.0}


def test_compute_fund_coverage_pct_empty():
    assert compute_fund_coverage_pct({}) == {}


def test_compute_look_through_exposure_combines_direct_and_fund_holding_into_one_row():
    # NVDA held directly ($1,000) AND via SPY ($10,000 position, 8% NVDA
    # weight = $800 look-through) -- must combine into ONE NVDA row.
    positions = [
        {"ticker": "NVDA", "market_value": 1_000.0},
        {"ticker": "SPY", "market_value": 10_000.0},
    ]
    fund_holdings = {"SPY": {"NVDA": 0.08, "AAPL": 0.07}}
    rows = compute_look_through_exposure(positions, fund_holdings)
    by_ticker = {r["ticker"]: r for r in rows}

    nvda = by_ticker["NVDA"]
    assert nvda["direct_value"] == pytest.approx(1_000.0)
    assert nvda["look_through_value"] == pytest.approx(800.0)
    assert nvda["combined_value"] == pytest.approx(1_800.0)
    assert nvda["via_funds"] == [{"fund_ticker": "SPY", "dollars": 800.0}]

    # AAPL only exists via SPY's look-through, never held directly.
    aapl = by_ticker["AAPL"]
    assert aapl["direct_value"] == 0.0
    assert aapl["look_through_value"] == pytest.approx(700.0)

    # SPY's own row is the undisclosed remainder: 1 - (0.08+0.07) = 0.85
    # of its $10,000 value, attributed to SPY itself.
    spy_row = by_ticker["SPY"]
    assert spy_row["direct_value"] == pytest.approx(8_500.0)
    assert spy_row["look_through_value"] == 0.0


def test_compute_look_through_exposure_conserves_total_dollars():
    positions = [
        {"ticker": "NVDA", "market_value": 1_000.0},
        {"ticker": "SPY", "market_value": 10_000.0},
        {"ticker": "GLD", "market_value": 2_000.0},
    ]
    fund_holdings = {"SPY": {"NVDA": 0.08, "AAPL": 0.07}, "GLD": {}}
    rows = compute_look_through_exposure(positions, fund_holdings)
    total_in = sum(p["market_value"] for p in positions)
    total_out = sum(r["combined_value"] for r in rows)
    assert total_out == pytest.approx(total_in, abs=0.01)


def test_compute_look_through_exposure_zero_holdings_fund_is_its_own_row():
    # GLD-like: a real fund with nothing disclosed (0 rows) -- must not
    # error, and its value stays attributed to itself.
    positions = [{"ticker": "GLD", "market_value": 5_000.0}]
    rows = compute_look_through_exposure(positions, fund_holdings={"GLD": {}})
    assert len(rows) == 1
    assert rows[0]["ticker"] == "GLD"
    assert rows[0]["combined_value"] == pytest.approx(5_000.0)


def test_compute_look_through_exposure_no_funds_is_pass_through():
    positions = [{"ticker": "AAPL", "market_value": 3_000.0}]
    rows = compute_look_through_exposure(positions, fund_holdings={})
    assert rows == [
        {
            "ticker": "AAPL", "direct_value": 3_000.0, "look_through_value": 0.0,
            "combined_value": 3_000.0, "combined_weight_pct": 100.0, "via_funds": [],
        }
    ]
