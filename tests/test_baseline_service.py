from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from services.baseline_service import (
    InsufficientHistoryError,
    _entry_prices,
    _forward_windows,
    compute_baseline_band,
)


def _make_ohlc(closes: list[float], start=date(2020, 1, 1)) -> pd.DataFrame:
    """
    Builds a minimal, well-formed daily OHLC frame from a list of closes —
    High/Low set a fixed +-1% around each close (Open == prior close) so
    tests can reason about excursions in terms of the closes alone unless a
    test overrides High/Low directly.
    """
    idx = [start + timedelta(days=i) for i in range(len(closes))]
    closes = np.array(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "Open": opens,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
        },
        index=pd.DatetimeIndex(idx),
    )


def _random_walk_ohlc(n: int, seed: int, daily_vol: float = 0.02, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, daily_vol, n)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    # Independent intraday range so High/Low aren't a fixed function of Close.
    intraday = rng.uniform(0.005, 0.02, n)
    highs = closes * (1 + intraday)
    lows = closes * (1 - intraday)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes},
        index=pd.DatetimeIndex(idx),
    )


# ---------------------------------------------------------------------------
# Window construction — the foundation everything else depends on.
# ---------------------------------------------------------------------------

def test_forward_windows_excludes_entry_day_and_aligns_correctly():
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    windows = _forward_windows(values, horizon=2)
    assert windows.shape == (4, 2)
    assert list(windows[0]) == [11.0, 12.0]  # entry i=0 -> looks at i+1, i+2
    assert list(windows[1]) == [12.0, 13.0]
    assert list(windows[3]) == [14.0, 15.0]  # entry i=3 -> last valid entry


def test_entry_prices_matches_forward_windows_count():
    close = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    entries = _entry_prices(close, horizon=2)
    windows = _forward_windows(close, horizon=2)
    assert len(entries) == len(windows)
    assert list(entries) == [10.0, 11.0, 12.0, 13.0]


# ---------------------------------------------------------------------------
# FR-03/04/04a — input validation and insufficient-history handling.
# ---------------------------------------------------------------------------

def test_rejects_non_positive_prices():
    df = _make_ohlc([100.0] * 50)
    df.iloc[10, df.columns.get_loc("Low")] = -5.0
    with pytest.raises(ValueError):
        compute_baseline_band(df, horizon_days=10)


def test_fr04_raises_with_shortfall_when_too_few_bars():
    df = _make_ohlc([100.0] * 20)  # need 10+2=12, but not enough for FR-04a either
    with pytest.raises(InsufficientHistoryError) as exc:
        compute_baseline_band(df, horizon_days=10)
    assert exc.value.bars_available == 20


def test_fr04a_raises_when_bars_pass_fr04_but_not_enough_independent_windows():
    # H=90: FR-04 only needs 92 bars; FR-04a needs 3 independent blocks (272).
    df = _make_ohlc(list(100 + np.sin(np.linspace(0, 10, 95))))
    with pytest.raises(InsufficientHistoryError) as exc:
        compute_baseline_band(df, horizon_days=90)
    assert exc.value.bars_available == 95
    assert exc.value.bars_required > 92  # stricter than the bare FR-04 floor


def test_sufficient_history_does_not_raise():
    df = _random_walk_ohlc(400, seed=1)
    result = compute_baseline_band(df, horizon_days=30)
    assert result["samples"] > 0


# ---------------------------------------------------------------------------
# Ladder shape and the median_path anchor.
# ---------------------------------------------------------------------------

def test_median_path_is_always_last_price_unmodified():
    df = _random_walk_ohlc(500, seed=2)
    result = compute_baseline_band(df, horizon_days=30)
    assert result["median_path"] == pytest.approx(result["last_price"])
    assert result["median_path_pct"] == 0.0


@pytest.mark.parametrize("horizon", [10, 30, 60])
@pytest.mark.parametrize("confidence", [0.90, 0.95])
def test_ladder_levels_are_correctly_ordered(horizon, confidence):
    df = _random_walk_ohlc(700, seed=3)
    result = compute_baseline_band(df, horizon_days=horizon, confidence=confidence)
    assert (
        result["floor"]
        <= result["accumulation_zone_hi"]
        <= result["median_path"]
        <= result["distribution_zone_lo"]
        <= result["ceiling"]
    )


def test_higher_confidence_widens_the_band():
    df = _random_walk_ohlc(700, seed=4)
    band_90 = compute_baseline_band(df, horizon_days=30, confidence=0.90)
    band_95 = compute_baseline_band(df, horizon_days=30, confidence=0.95)
    assert band_95["floor"] <= band_90["floor"]
    assert band_95["ceiling"] >= band_90["ceiling"]


# ---------------------------------------------------------------------------
# US-06 — calibration diagnostics.
# ---------------------------------------------------------------------------

def test_breach_rate_is_close_to_expected_on_its_own_fitting_data():
    # The floor is defined as the (1-confidence) percentile of this exact
    # distribution, so checking breach against the same data it was fit on
    # should reconstruct close to (1-confidence) — this is the core
    # self-consistency the calibration mechanism depends on.
    df = _random_walk_ohlc(1500, seed=5, daily_vol=0.015)
    result = compute_baseline_band(df, horizon_days=30, confidence=0.90, half_life=None)
    assert abs(result["breach_rate_full"] - result["expected_breach"]) < 0.03
    assert result["calibration_warning"] is False


def test_effective_samples_uses_nonoverlapping_block_approximation():
    df = _random_walk_ohlc(700, seed=6)
    result = compute_baseline_band(df, horizon_days=30)
    assert result["effective_samples"] == result["samples"] // 30
    assert result["effective_samples"] < result["samples"]  # overlap correction actually reduces it


def test_recency_weighting_changes_the_result():
    # Calm first half, much more volatile second half. A short half-life
    # concentrates almost all weight on a small, recent slice of entries —
    # which can land shallower *or* deeper than the full-history percentile
    # depending on the specific draw (fewer effective samples means fewer
    # chances to have captured an extreme tail window, regardless of the
    # underlying regime's true volatility) — so this checks weighting has a
    # real, material effect on the result (FR-15), not a specific direction.
    rng = np.random.default_rng(7)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 500)))
    wild = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.05, 500)))
    closes = np.concatenate([calm, wild])
    df = _make_ohlc(list(closes))
    unweighted = compute_baseline_band(df, horizon_days=30, half_life=None)
    weighted = compute_baseline_band(df, horizon_days=30, half_life=30)
    assert abs(weighted["floor_pct"] - unweighted["floor_pct"]) > 2.0


# ---------------------------------------------------------------------------
# AC-04.1/04.2 — shape metrics, and the sqrt method.
# ---------------------------------------------------------------------------

def test_shape_metrics_are_within_sane_bounds():
    df = _random_walk_ohlc(700, seed=8)
    result = compute_baseline_band(df, horizon_days=30)
    assert result["rr_ratio"] is None or result["rr_ratio"] > 0
    assert 0.0 <= result["upside_first_rate"] <= 1.0


def test_sqrt_method_runs_and_orders_correctly():
    df = _random_walk_ohlc(700, seed=9)
    result = compute_baseline_band(df, horizon_days=30, method="sqrt")
    assert result["method"] == "sqrt"
    assert result["floor"] <= result["median_path"] <= result["ceiling"]


def test_invalid_parameters_raise_value_error():
    df = _random_walk_ohlc(700, seed=10)
    with pytest.raises(ValueError):
        compute_baseline_band(df, horizon_days=45)  # not in HORIZONS
    with pytest.raises(ValueError):
        compute_baseline_band(df, confidence=1.0)  # must be < 1.0
    with pytest.raises(ValueError):
        compute_baseline_band(df, method="sum")  # FR-14: not a selectable method
