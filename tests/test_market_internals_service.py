import numpy as np
import pandas as pd
import pytest

from services.market_internals_service import (
    HYSTERESIS_SESSIONS,
    apply_hysteresis,
    compute_composite_score,
    compute_internals_score,
    map_regime,
    run_forward_return_backtest,
)


def _dates(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def _flat_internals(n, breadth=50.0, vix=18.0, vix3m=19.0, xly_xlp=1.0, hyg_ief=1.0, rsp_spy=1.0):
    idx = _dates(n)
    return pd.DataFrame(
        {
            "breadth_50dma": breadth,
            "vix": vix,
            "vix3m": vix3m,
            "xly_xlp": xly_xlp,
            "hyg_ief": hyg_ief,
            "rsp_spy": rsp_spy,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# compute_internals_score
# ---------------------------------------------------------------------------

def test_score_is_nan_before_zscore_window_fills():
    df = _flat_internals(100)
    score = compute_internals_score(df)
    assert score.iloc[:249].isna().all() if len(score) > 249 else score.isna().all()


def test_flat_inputs_score_near_zero_once_window_fills():
    df = _flat_internals(300)
    score = compute_internals_score(df)
    # Flat series -> zero variance -> z-score NaN via std=0 guard, so score
    # is NaN throughout for a perfectly flat input (no distribution to
    # score against) rather than an ill-defined divide-by-zero unlike NaN.
    assert score.iloc[-1] is None or pd.isna(score.iloc[-1]) or abs(score.iloc[-1]) < 1e-6


def test_high_vix_pulls_score_down():
    n = 400
    idx = _dates(n)
    rng = np.random.default_rng(0)
    df = _flat_internals(n)
    df["vix"] = 15 + rng.normal(0, 2, n)
    df.index = idx

    calm = compute_internals_score(df).iloc[-1]

    df_spike = df.copy()
    df_spike.iloc[-1, df_spike.columns.get_loc("vix")] = 60.0  # extreme spike on the last day
    spike = compute_internals_score(df_spike).iloc[-1]

    assert spike < calm


def test_rising_breadth_pulls_score_up():
    n = 400
    idx = _dates(n)
    rng = np.random.default_rng(1)
    df = _flat_internals(n)
    df["breadth_50dma"] = 50 + rng.normal(0, 3, n)
    df.index = idx

    baseline = compute_internals_score(df).iloc[-1]

    df_strong = df.copy()
    df_strong.iloc[-1, df_strong.columns.get_loc("breadth_50dma")] = 95.0
    strong = compute_internals_score(df_strong).iloc[-1]

    assert strong > baseline


def test_score_is_clipped_to_range():
    n = 400
    rng = np.random.default_rng(2)
    df = _flat_internals(n)
    df["vix"] = 15 + rng.normal(0, 2, n)
    df.index = _dates(n)
    df.iloc[-1, df.columns.get_loc("vix")] = 500.0  # absurd outlier
    score = compute_internals_score(df)
    assert -100 <= score.iloc[-1] <= 100


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------

def test_composite_internals_only_uses_full_weight():
    result = compute_composite_score(internals_score=50.0)
    assert result["mds"] == 50.0
    assert result["data_completeness"] == pytest.approx(1 / 3, abs=1e-3)


def test_composite_all_pillars_present():
    result = compute_composite_score(internals_score=50.0, news_score=50.0, earnings_score=50.0)
    assert result["mds"] == 50.0
    assert result["data_completeness"] == 1.0


def test_composite_no_pillars_returns_none():
    result = compute_composite_score(internals_score=None)
    assert result["mds"] is None
    assert result["data_completeness"] == 0.0


def test_composite_conflict_flag_fires_on_large_disagreement():
    result = compute_composite_score(internals_score=80.0, news_score=-10.0)
    assert result["conflict_flag"] is True


def test_composite_no_conflict_when_pillars_agree():
    result = compute_composite_score(internals_score=40.0, news_score=30.0)
    assert result["conflict_flag"] is False


# ---------------------------------------------------------------------------
# map_regime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mds,expected",
    [(80, "Risk-On"), (40, "Constructive"), (0, "Neutral"), (-40, "Cautious"), (-80, "Risk-Off")],
)
def test_map_regime_bands(mds, expected):
    assert map_regime(mds) == expected


def test_map_regime_none_for_none_input():
    assert map_regime(None) is None


# ---------------------------------------------------------------------------
# apply_hysteresis
# ---------------------------------------------------------------------------

def test_hysteresis_requires_consecutive_sessions():
    # Single-day blip back to Neutral should NOT flip the confirmed regime.
    regimes = pd.Series(
        ["Constructive"] * 5 + ["Neutral"] + ["Constructive"] * 5,
        index=_dates(11),
    )
    confirmed = apply_hysteresis(regimes)
    assert (confirmed == "Constructive").all()


def test_hysteresis_confirms_after_enough_sessions():
    regimes = pd.Series(
        ["Constructive"] * 5 + ["Risk-Off"] * HYSTERESIS_SESSIONS + ["Risk-Off"] * 3,
        index=_dates(5 + HYSTERESIS_SESSIONS + 3),
    )
    confirmed = apply_hysteresis(regimes)
    assert confirmed.iloc[-1] == "Risk-Off"
    assert (confirmed.iloc[:5] == "Constructive").all()


# ---------------------------------------------------------------------------
# run_forward_return_backtest
# ---------------------------------------------------------------------------

def test_backtest_reports_stats_per_regime_and_horizon():
    n = 300
    idx = _dates(n)
    rng = np.random.default_rng(3)
    scores = pd.Series(rng.uniform(-100, 100, n), index=idx)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)

    result = run_forward_return_backtest(scores, price, horizons=(1, 5))
    assert set(result.keys()) == {1, 5}
    for horizon_result in result.values():
        assert set(horizon_result.keys()) == {
            "Risk-On", "Constructive", "Neutral", "Cautious", "Risk-Off",
        }
        for bucket in horizon_result.values():
            assert bucket["n"] >= 0
            if bucket["n"] > 0:
                assert bucket["mean_pct"] is not None
                assert 0.0 <= bucket["hit_rate"] <= 1.0


def test_backtest_handles_empty_regime_bucket_gracefully():
    # All scores land in one regime -> other buckets must report n=0, not error.
    idx = _dates(50)
    scores = pd.Series([90.0] * 50, index=idx)  # always Risk-On
    price = pd.Series(np.linspace(100, 110, 50), index=idx)
    result = run_forward_return_backtest(scores, price, horizons=(1,))
    assert result[1]["Risk-Off"]["n"] == 0
    assert result[1]["Risk-Off"]["mean_pct"] is None
