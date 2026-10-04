import numpy as np
import pandas as pd
import pytest

from services.strategy_robustness import deflated_sharpe, sensitivity, walk_forward


def _series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_walk_forward_rolls_six_month_windows_after_the_first_two_years():
    n = 504 + 126 * 3
    strat = _series([0.1] * n)
    basket = _series([0.0] * n)
    wf = walk_forward(strat, basket, train=504, test=126)
    assert wf["test_windows"] == 3
    assert wf["beat_basket_windows"] == 3
    assert wf["beat_basket_pct"] == pytest.approx(100.0)
    assert wf["windows"][0]["strategy_pct"] > wf["windows"][0]["basket_pct"]


def test_walk_forward_counts_a_losing_window_as_not_beating_the_basket():
    n = 504 + 126 * 2
    strat = _series([-0.1] * n)
    basket = _series([0.0] * n)
    wf = walk_forward(strat, basket, train=504, test=126)
    assert wf["beat_basket_windows"] == 0
    assert wf["beat_basket_pct"] == pytest.approx(0.0)


def test_sensitivity_moves_each_threshold_down_and_up_by_the_step():
    # Sharpe falls as the threshold rises: the swing is the spread across the three settings.
    def run(params):
        return 1.0 - params["threshold"] / 100
    s = sensitivity(run, {"threshold": 50.0}, step=0.10)
    row = s["rows"][0]
    values = [c["value"] for c in row["cells"]]
    assert values == pytest.approx([45.0, 50.0, 55.0])
    assert row["swing"] == pytest.approx(0.1, abs=1e-3)
    assert s["base_sharpe"] == pytest.approx(0.5)


def test_deflated_sharpe_falls_as_more_variants_are_tried():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.08, 1.0, 1000))  # a modest positive edge
    single = deflated_sharpe(returns, [0.05])
    many = deflated_sharpe(returns, list(rng.normal(0.05, 0.05, 50)))
    assert single["probability"] > many["probability"]
    assert many["variants"] == 50


def test_deflated_sharpe_with_no_edge_is_near_a_coin_flip_or_below():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 1.0, 800))
    result = deflated_sharpe(returns, list(rng.normal(0.0, 0.05, 30)))
    assert result["probability"] is not None
    assert result["probability"] < 0.8
