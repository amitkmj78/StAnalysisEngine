import numpy as np
import pandas as pd

from services.regime_validation import MIN_DAYS, divergence_series, forward_risk, validate


def _calm_then_wild(n_each=120, seed=1):
    rng = np.random.default_rng(seed)
    calm = 100 * np.cumprod(1 + rng.normal(0.0004, 0.004, n_each))
    wild = calm[-1] * np.cumprod(1 + rng.normal(0.0, 0.03, n_each))
    values = np.concatenate([calm, wild])
    return pd.Series(values, index=pd.bdate_range("2025-01-01", periods=len(values)))


def test_forward_vol_is_higher_in_the_wild_stretch():
    spy = _calm_then_wild()
    risk = forward_risk(spy)
    calm_vol = risk.iloc[:60]["fwd_vol_pct"].mean()
    wild_vol = risk.iloc[150:]["fwd_vol_pct"].mean()
    assert wild_vol > 3 * calm_vol


def test_worst_drawdown_is_zero_or_negative_and_measured_forward():
    spy = pd.Series([100.0, 90.0, 95.0] + [96.0] * 30, index=pd.bdate_range("2025-01-01", periods=33))
    risk = forward_risk(spy, horizon=3)
    assert round(risk.iloc[0]["fwd_worst_dd_pct"], 2) == -10.0


def test_label_test_passes_when_risk_labels_match_the_wild_stretch():
    spy = _calm_then_wild()
    dates = spy.index
    labels = pd.Series(["Risk-On"] * 120 + ["Cautious"] * 120, index=dates)
    flags = pd.Series(False, index=dates)
    result = validate(labels, spy, flags)
    assert result["label_test"]["result"] == "pass"


def test_insufficient_data_when_a_group_is_missing_not_a_pass():
    spy = _calm_then_wild()
    labels = pd.Series(["Neutral"] * len(spy), index=spy.index)  # no Risk-On and no Risk-Off days
    result = validate(labels, spy, pd.Series(False, index=spy.index))
    assert result["label_test"]["result"] == "insufficient data"
    assert result["label_groups"]["Risk-On"]["days"] < MIN_DAYS


def test_divergence_series_matches_the_banner_rule():
    spy = pd.Series(np.linspace(100, 120, 80), index=pd.bdate_range("2025-01-01", periods=80))
    breadth = pd.Series(30.0, index=spy.index)
    flags = divergence_series(spy, breadth)
    assert bool(flags.iloc[-1]) is True
    assert bool(flags.iloc[10]) is False  # not enough history for the 50-day average yet
