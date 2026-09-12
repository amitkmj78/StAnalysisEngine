from unittest.mock import patch

import services.momentum_backtest_service as mbs


def _clear_cache():
    mbs._backtest_cache.clear()
    mbs._backtest_cache_ts.clear()


def test_none_result_is_not_cached():
    """Regression test: a not-enough-data (None) result must not be
    remembered for the full 6h TTL -- production hit exactly this, where
    a transient yf.download hiccup for one parameter combination (30-day
    lookback, 10-day hold, 3 years) got cached as permanently
    unbacktestable even though the underlying data issue was momentary."""
    _clear_cache()
    with patch.object(mbs, "_compute_backtest_momentum_ranking", return_value=None) as mock_compute:
        first = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 10, 5.0, 0.0, 30.0, 0.0)
        second = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 10, 5.0, 0.0, 30.0, 0.0)

    assert first is None
    assert second is None
    # Not cached -- the second call must hit the real function again,
    # not a remembered None.
    assert mock_compute.call_count == 2


def test_real_result_is_cached():
    """The other half of the same fix: a real result IS cached (it's
    expensive to compute and doesn't need to be real-time) -- a second
    call with identical params shouldn't recompute."""
    _clear_cache()
    fake_result = {"asset_type": "Stock", "universe": "All", "num_periods": 10}
    with patch.object(mbs, "_compute_backtest_momentum_ranking", return_value=fake_result) as mock_compute:
        first = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 10, 5.0, 0.0, 30.0, 0.0)
        second = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 10, 5.0, 0.0, 30.0, 0.0)

    assert first == fake_result
    assert second == fake_result
    assert mock_compute.call_count == 1


def test_different_params_are_cached_separately():
    _clear_cache()
    with patch.object(
        mbs, "_compute_backtest_momentum_ranking", side_effect=[{"horizon_days": 10}, {"horizon_days": 30}]
    ) as mock_compute:
        ten_day = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 10, 5.0, 0.0, 30.0, 0.0)
        thirty_day = mbs.backtest_momentum_ranking("Stock", "All", 30, 5, 3, 30, 5.0, 0.0, 30.0, 0.0)

    assert ten_day == {"horizon_days": 10}
    assert thirty_day == {"horizon_days": 30}
    assert mock_compute.call_count == 2
