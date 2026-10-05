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
    oos = [c["oos_excess_return_pct"] for c in result["candidates"]]
    assert oos == sorted(oos, reverse=True)
    assert result["variants_tried"] == len(TEMPLATES)
    assert "not recommendations" in result["note"]


def test_warnings_describe_the_direction_of_the_gap():
    assert "lost on later ones" in " ".join(_warnings(is_excess=5.0, oos_excess=-2.0))
    assert "Did not beat" in " ".join(_warnings(is_excess=-1.0, oos_excess=-2.0))
    assert _warnings(is_excess=3.0, oos_excess=4.0) == []


def test_pick_sample_is_reproducible_for_a_seed():
    universe = [f"S{i:03d}" for i in range(500)]
    assert pick_sample(universe, seed=7) == pick_sample(universe, seed=7)
    assert len(pick_sample(universe, seed=7)) == 20
