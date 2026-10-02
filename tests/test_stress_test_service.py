"""
Scenario and Stress Tests (STR-1..3). No live network -- every test
builds synthetic price series and monkeypatches the module's
data-fetching functions, matching tests/test_portfolio_health_service.py's
established convention.
"""

import pandas as pd
import pytest

import services.portfolio_health_service as phs
import services.stress_test_service as sts
from services.stress_test_service import (
    HISTORICAL_REPLAYS,
    SHOCK_PRESETS,
    run_custom_scenario,
    run_historical_replay,
    run_preset_shock,
)


def _prices(values, start="2023-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# run_preset_shock (STR-1 presets + STR-3 disclosures)
# ---------------------------------------------------------------------------


def test_run_preset_shock_market_down_10_uses_portfolio_beta(monkeypatch):
    bench_moves = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.005, 0.01] * 5
    bench_prices = [100.0]
    for m in bench_moves:
        bench_prices.append(bench_prices[-1] * (1 + m))
    stock_prices = [50.0]
    for m in bench_moves:
        stock_prices.append(stock_prices[-1] * (1 + 1.5 * m))

    def fake_history(ticker, period, auto_adjust=True):
        if ticker == "SPY":
            return pd.DataFrame({"Close": _prices(bench_prices)})
        return pd.DataFrame({"Close": _prices(stock_prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    positions = [{"ticker": "LEVERED", "market_value": 10_000.0}]
    result = run_preset_shock(positions, "market_down_10")

    assert result["beta"] == pytest.approx(1.5, abs=0.05)
    assert result["estimated_pct_impact"] == pytest.approx(1.5 * -10.0, abs=0.5)
    assert result["estimated_dollar_impact"] == pytest.approx(10_000.0 * (1.5 * -10.0) / 100.0, abs=50.0)
    assert result["total_market_value"] == 10_000.0
    assert result["benchmark_ticker"] == "SPY"
    assert "SPY" in result["method"]


def test_run_preset_shock_includes_method_and_benchmark_for_every_preset():
    # STR-3 regression guard: a future preset added without a method
    # string must fail this test.
    for key, preset in SHOCK_PRESETS.items():
        result_shape = run_preset_shock([], key)
        assert result_shape["method"] == preset["method"]
        assert isinstance(result_shape["method"], str) and result_shape["method"].strip()
        assert result_shape["benchmark_ticker"] == preset["benchmark_ticker"]
        assert result_shape["shock_pct"] == preset["shock_pct"]


def test_run_preset_shock_rates_preset_shock_pct_is_negative_seventeen():
    # Pinned so a future "improvement" can't silently change the implied
    # TLT move without also updating the method text that discloses it.
    assert SHOCK_PRESETS["rates_up_1pct"]["shock_pct"] == -17.0
    assert SHOCK_PRESETS["rates_up_1pct"]["benchmark_ticker"] == "TLT"


def test_run_preset_shock_no_positions_returns_null_impact_not_zero():
    result = run_preset_shock([], "market_down_10")
    assert result["estimated_pct_impact"] is None
    assert result["estimated_dollar_impact"] is None
    assert result["total_market_value"] == 0.0


def test_run_preset_shock_unknown_preset_key_raises():
    with pytest.raises(KeyError):
        run_preset_shock([{"ticker": "AAPL", "market_value": 100.0}], "not_a_real_preset")


# ---------------------------------------------------------------------------
# run_historical_replay
# ---------------------------------------------------------------------------


def test_run_historical_replay_actual_return_matches_price_series(monkeypatch):
    def fake_range(ticker, start, end, auto_adjust=True):
        return pd.DataFrame({"Close": [100.0, 110.0]})

    monkeypatch.setattr(sts, "get_cached_history_range", fake_range)

    positions = [{"ticker": "AAPL", "market_value": 1_000.0}]
    result = run_historical_replay(positions, "2008_gfc")

    assert result["holdings"][0]["estimated_pct_impact"] == pytest.approx(10.0, abs=0.001)
    assert result["holdings"][0]["estimated_dollar_impact"] == pytest.approx(100.0, abs=0.01)
    assert result["holdings"][0]["method_used"] == "actual"
    assert result["excluded_holdings"] == []
    assert result["window_start"] == HISTORICAL_REPLAYS["2008_gfc"]["start"]
    assert result["window_end"] == HISTORICAL_REPLAYS["2008_gfc"]["end"]


def test_run_historical_replay_excludes_ticker_with_no_data_and_discloses_it(monkeypatch):
    def fake_range(ticker, start, end, auto_adjust=True):
        if ticker == "NEWCO":
            return pd.DataFrame()
        return pd.DataFrame({"Close": [100.0, 90.0]})

    monkeypatch.setattr(sts, "get_cached_history_range", fake_range)

    positions = [
        {"ticker": "OLDCO", "market_value": 1_000.0},
        {"ticker": "NEWCO", "market_value": 500.0},
    ]
    result = run_historical_replay(positions, "2020_covid")

    newco_row = next(h for h in result["holdings"] if h["ticker"] == "NEWCO")
    assert newco_row["method_used"] == "excluded_no_history_for_window"
    assert newco_row["estimated_dollar_impact"] is None
    assert result["excluded_holdings"] == ["NEWCO"]
    assert "NEWCO" in result["method"]


def test_run_historical_replay_total_pct_is_dollar_weighted_not_simple_average(monkeypatch):
    def fake_range(ticker, start, end, auto_adjust=True):
        if ticker == "BIG":
            return pd.DataFrame({"Close": [100.0, 90.0]})  # -10%
        return pd.DataFrame({"Close": [100.0, 50.0]})  # -50%

    monkeypatch.setattr(sts, "get_cached_history_range", fake_range)

    positions = [
        {"ticker": "BIG", "market_value": 9_000.0},
        {"ticker": "SMALL", "market_value": 1_000.0},
    ]
    result = run_historical_replay(positions, "2022_bear")

    # Dollar-weighted: 9000*-0.10 + 1000*-0.50 = -900 - 500 = -1400, / 10000 = -14%.
    # Simple average would be (-10 + -50)/2 = -30%, clearly different.
    assert result["estimated_pct_impact"] == pytest.approx(-14.0, abs=0.01)
    assert result["estimated_dollar_impact"] == pytest.approx(-1400.0, abs=0.01)


def test_run_historical_replay_empty_positions_returns_null_impact():
    result = run_historical_replay([], "2008_gfc")
    assert result["estimated_pct_impact"] is None
    assert result["estimated_dollar_impact"] is None
    assert result["holdings"] == []


def test_run_historical_replay_unknown_key_raises():
    with pytest.raises(KeyError):
        run_historical_replay([{"ticker": "AAPL", "market_value": 100.0}], "not_a_real_replay")


def test_historical_replays_dates_match_documented_windows():
    # STR-3 requires the *stated* window to be the *actual* window used --
    # pin the six date strings so an accidental typo can't silently ship.
    assert HISTORICAL_REPLAYS["2008_gfc"]["start"] == "2008-09-01"
    assert HISTORICAL_REPLAYS["2008_gfc"]["end"] == "2009-03-09"
    assert HISTORICAL_REPLAYS["2020_covid"]["start"] == "2020-02-19"
    assert HISTORICAL_REPLAYS["2020_covid"]["end"] == "2020-03-23"
    assert HISTORICAL_REPLAYS["2022_bear"]["start"] == "2022-01-03"
    assert HISTORICAL_REPLAYS["2022_bear"]["end"] == "2022-10-12"


# ---------------------------------------------------------------------------
# run_custom_scenario (STR-2)
# ---------------------------------------------------------------------------


def test_run_custom_scenario_sector_component_only_affects_that_sector():
    positions = [
        {"ticker": "TECH1", "market_value": 10_000.0},
        {"ticker": "FIN1", "market_value": 5_000.0},
    ]
    sector_by_ticker = {"TECH1": "Technology", "FIN1": "Financials"}
    components = [{"kind": "sector", "sector": "Technology", "shock_pct": -15.0, "label": "Tech -15%"}]

    result = run_custom_scenario(positions, sector_by_ticker, components)

    assert result["total_estimated_dollar_impact"] == pytest.approx(10_000.0 * -0.15, abs=0.01)
    assert result["components"][0]["estimated_dollar_impact"] == pytest.approx(-1500.0, abs=0.01)
    # Financials holding's value never enters the sector-shock computation.
    assert "FIN1" not in result["components"][0]["method"]


def test_run_custom_scenario_factor_component_uses_blended_beta(monkeypatch):
    bench_moves = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.008, -0.012, 0.005, 0.01] * 5
    bench_prices = [100.0]
    for m in bench_moves:
        bench_prices.append(bench_prices[-1] * (1 + m))
    stock_prices = [50.0]
    for m in bench_moves:
        stock_prices.append(stock_prices[-1] * (1 + 1.5 * m))

    def fake_history(ticker, period, auto_adjust=True):
        if ticker == "SPY":
            return pd.DataFrame({"Close": _prices(bench_prices)})
        return pd.DataFrame({"Close": _prices(stock_prices)})

    monkeypatch.setattr(phs, "get_cached_history", fake_history)

    positions = [{"ticker": "LEVERED", "market_value": 10_000.0}]
    components = [{"kind": "factor", "benchmark_ticker": "SPY", "shock_pct": -10.0, "label": "Market -10%"}]

    result = run_custom_scenario(positions, {}, components)

    expected_pct = 1.5 * -10.0
    assert result["components"][0]["estimated_dollar_impact"] == pytest.approx(
        10_000.0 * expected_pct / 100.0, abs=50.0
    )


def test_run_custom_scenario_combines_components_additively():
    positions = [{"ticker": "TECH1", "market_value": 10_000.0}]
    sector_by_ticker = {"TECH1": "Technology"}
    components = [
        {"kind": "sector", "sector": "Technology", "shock_pct": -10.0, "label": "Tech -10%"},
        {"kind": "sector", "sector": "Technology", "shock_pct": -5.0, "label": "Tech -5% more"},
    ]
    result = run_custom_scenario(positions, sector_by_ticker, components)

    c0 = result["components"][0]["estimated_dollar_impact"]
    c1 = result["components"][1]["estimated_dollar_impact"]
    assert result["total_estimated_dollar_impact"] == pytest.approx(c0 + c1, abs=0.01)


def test_run_custom_scenario_unknown_component_kind_raises():
    with pytest.raises(ValueError):
        run_custom_scenario(
            [{"ticker": "AAPL", "market_value": 100.0}], {},
            [{"kind": "not_a_real_kind", "shock_pct": -5.0, "label": "x"}],
        )


def test_run_custom_scenario_empty_components_returns_zero_impact_not_null():
    # An explicitly empty scenario is a stated $0, not an unknown impact --
    # distinct from "no positions at all".
    positions = [{"ticker": "AAPL", "market_value": 1_000.0}]
    result = run_custom_scenario(positions, {}, [])
    assert result["total_estimated_dollar_impact"] == 0.0
    assert result["total_estimated_pct_impact"] == 0.0
    assert result["components"] == []
