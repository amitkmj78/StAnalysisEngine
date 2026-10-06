import numpy as np
import pandas as pd
import pytest

import strategy_sdk


def _prices(seed, n=800):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx)


def _fake_history(monkeypatch):
    data = {"AAA": _prices(1), "BBB": _prices(2), "SPY": _prices(3)}
    monkeypatch.setattr(strategy_sdk, "get_cached_history", lambda t, period, adj, interval: data.get(t, pd.DataFrame()))


def test_the_sdk_runs_the_same_engine_and_returns_the_explanation(monkeypatch):
    _fake_history(monkeypatch)
    result = strategy_sdk.backtest(
        ["AAA", "BBB"],
        [{"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0}],
        [{"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": 0}],
        trailing_stop_pct=15,
    )
    assert {"strategy", "basket", "benchmark_spy", "checks", "explanation"} <= set(result)
    assert result["explanation"]["bottom_line"]


def test_a_test_without_a_protective_exit_is_refused(monkeypatch):
    _fake_history(monkeypatch)
    with pytest.raises(ValueError, match="protective exit"):
        strategy_sdk.backtest(["AAA", "BBB"], [{"field": "rsi_14", "op": "<", "value": 40}], [])


def test_a_ticker_with_no_prices_is_named_in_the_error(monkeypatch):
    _fake_history(monkeypatch)
    with pytest.raises(ValueError, match="No price history for ZZZ"):
        strategy_sdk.backtest(["AAA", "ZZZ"], [{"field": "rsi_14", "op": "<", "value": 40}], [], stop_loss_pct=8)
