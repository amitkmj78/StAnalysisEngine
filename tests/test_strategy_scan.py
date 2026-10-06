import numpy as np
import pandas as pd

from services.strategy_engine import feature_frame
from services.strategy_scan import TEMPLATES, _warnings, pick_sample, scan


def _prices(seed, n=420):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx)


def test_scan_ranks_every_template_and_reports_the_variant_count():
    frames = {f"T{i}": feature_frame(_prices(i)) for i in range(3)}
    result = scan(frames, _prices(99)["Close"])
    assert len(result["candidates"]) == len(TEMPLATES)
    assert [c["rank"] for c in result["candidates"]] == list(range(1, len(TEMPLATES) + 1))
    oos = [c["oos_return_after_costs_pct"] for c in result["candidates"]]
    assert oos == sorted(oos, reverse=True)
    assert result["variants_tried"] == len(TEMPLATES)
    assert "not recommendations" in result["note"]
    assert set(result["groups"]) == {"short_term", "long_term"}
    assert result["groups"]["short_term"]["message"] or result["groups"]["short_term"]["candidates"]


def test_warnings_describe_the_direction_of_the_gap():
    assert "lost on later ones" in " ".join(_warnings(is_excess=5.0, oos_excess=-2.0))
    assert "Did not beat" in " ".join(_warnings(is_excess=-1.0, oos_excess=-2.0))
    assert _warnings(is_excess=3.0, oos_excess=4.0) == []


def test_pick_sample_is_reproducible_for_a_seed():
    universe = [f"S{i:03d}" for i in range(500)]
    assert pick_sample(universe, seed=7) == pick_sample(universe, seed=7)
    assert len(pick_sample(universe, seed=7)) == 20


def test_the_card_shows_the_template_and_holding_returns_and_the_difference():
    from services.strategy_scan import _profile

    row = {"holding_class": "short", "avg_hold_days": 18.0, "trades_oos": 130,
           "oos_return_after_costs_pct": 22.3, "oos_holding_return_pct": 58.0, "oos_difference_pts": -35.7}
    text = _profile(row)
    assert "+22.3% after costs vs +58.0% for holding (-35.7 pts)" in text


def test_the_did_not_beat_verdict_agrees_with_the_difference_sign():
    from services.strategy_scan import _warnings

    assert any("Did not beat holding" in w for w in _warnings(0.0, -35.7))
    assert not any("Did not beat holding" in w for w in _warnings(0.0, 4.2))


def test_a_template_passes_on_return_or_on_risk_with_no_deeper_drawdown():
    from services.strategy_scan import _pass_test

    assert _pass_test(5.0, 0.2, 0.5, -30.0, -20.0) == "return"
    assert _pass_test(-5.0, 1.0, 0.5, -15.0, -20.0) == "risk"   # lost on return, better Sharpe, shallower drawdown
    assert _pass_test(-5.0, 1.0, 0.5, -25.0, -20.0) is None      # better Sharpe but a deeper drawdown does not pass
    assert _pass_test(-5.0, 0.2, 0.5, -15.0, -20.0) is None      # shallower drawdown but worse Sharpe does not pass


def test_the_pass_label_and_banner_rule_agree_with_the_test():
    from services.strategy_scan import _warnings

    assert not any("Did not beat" in w for w in _warnings(0.0, -5.0, "risk"))
    assert any("Did not beat" in w for w in _warnings(0.0, -5.0, None))


def test_calmar_is_return_over_drawdown():
    from services.strategy_scan import _calmar

    assert _calmar(20.0, -10.0) == 2.0
    assert _calmar(20.0, 0.0) is None
