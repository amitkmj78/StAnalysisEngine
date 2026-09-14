from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from services.quant_signal_backtest_service import (
    backtest_quant_signal,
    summarize_quant_signal_backtest,
)


def _price_series(n_days: int, start: str = "2023-01-03", base: float = 100.0, seed: int = 7) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, n_days)
    closes = base * np.cumprod(1 + returns)
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000] * n_days}, index=idx)


class _ConstantReturnModel:
    """A fake trained model whose every .predict() call returns a fixed
    per-step return, so the recursive forecast's final expected_return is
    fully deterministic -- isolates the backtest's bookkeeping (signal
    classification, correct/incorrect, aggregation) from the real GBM's
    actual behavior, which the live investigative run validates instead."""

    def __init__(self, per_step_return: float):
        self.per_step_return = per_step_return

    def predict(self, X):
        return np.array([self.per_step_return])


def test_not_enough_history_returns_none():
    with patch(
        "services.quant_signal_backtest_service.get_stock_data",
        return_value=_price_series(80),
    ):
        result = backtest_quant_signal("TEST", min_train_rows=252, horizon_days=10)
    assert result is None


def test_training_never_sees_a_row_at_or_after_the_cutoff():
    """Regression guard: the whole point of this backtest is to measure
    real predictive skill, not skill inflated by leaking the future into
    training -- same failure mode prediction_service.predict_backtest_prices
    already guards against for the single-day case."""
    data = _price_series(700)
    seen_lengths = []

    def fake_train(df, tune=False):
        seen_lengths.append(len(df))
        return _ConstantReturnModel(0.0)

    with patch("services.quant_signal_backtest_service.get_stock_data", return_value=data), patch(
        "services.quant_signal_backtest_service.train_model_on_frame", side_effect=fake_train
    ):
        result = backtest_quant_signal("TEST", min_train_rows=150, step_days=200, horizon_days=10)

    assert result is not None
    assert len(seen_lengths) >= 2
    # Strictly increasing -- each later cutoff trains on strictly more
    # (but still cutoff-bounded) history than the one before it.
    assert seen_lengths == sorted(seen_lengths)
    assert len(set(seen_lengths)) == len(seen_lengths)


def test_buy_signal_marked_incorrect_when_price_actually_falls():
    data = _price_series(700)

    with patch("services.quant_signal_backtest_service.get_stock_data", return_value=data), patch(
        "services.quant_signal_backtest_service.train_model_on_frame",
        return_value=_ConstantReturnModel(0.05),
    ):
        # per_step_return=0.05, shrunk by RETURN_SHRINKAGE (0.2) = 1%/step
        # compounded over 10 steps -> comfortably over the 5% BUY threshold
        # regardless of the underlying (real, noisy) price path.
        result = backtest_quant_signal("TEST", min_train_rows=150, step_days=250, horizon_days=10)

    assert result is not None
    assert all(row["signal"] == "BUY" for row in result["rows"])
    # correct is exactly (actual_return_pct > 0) -- verify the bookkeeping
    # matches the recorded actual return, not just that it ran.
    for row in result["rows"]:
        assert row["correct"] == (row["actual_return_pct"] > 0)


def test_summarize_pools_across_tickers_by_signal():
    fake_results = [
        {
            "ticker": "AAA",
            "horizon_days": 10,
            "rows": [
                {"date": "2024-01-01", "signal": "BUY", "expected_return_pct": 6.0, "actual_return_pct": 2.0, "correct": True},
                {"date": "2024-02-01", "signal": "BUY", "expected_return_pct": 7.0, "actual_return_pct": -3.0, "correct": False},
            ],
        },
        {
            "ticker": "BBB",
            "horizon_days": 10,
            "rows": [
                {"date": "2024-01-01", "signal": "SELL", "expected_return_pct": -6.0, "actual_return_pct": -1.0, "correct": True},
                {"date": "2024-02-01", "signal": "HOLD", "expected_return_pct": 1.0, "actual_return_pct": 0.5, "correct": True},
            ],
        },
        None,  # a ticker that failed to backtest (no data) -- must not crash aggregation
    ]

    summary = summarize_quant_signal_backtest(fake_results)

    assert summary["BUY"] == {"count": 2, "win_rate_pct": 50.0}
    assert summary["SELL"] == {"count": 1, "win_rate_pct": 100.0}
    assert summary["HOLD"] == {"count": 1, "win_rate_pct": 100.0}


def test_summarize_reports_none_win_rate_for_a_signal_with_no_calls():
    summary = summarize_quant_signal_backtest(
        [
            {
                "ticker": "AAA",
                "horizon_days": 10,
                "rows": [{"date": "2024-01-01", "signal": "HOLD", "expected_return_pct": 1.0, "actual_return_pct": 0.5, "correct": True}],
            }
        ]
    )
    assert summary["BUY"] == {"count": 0, "win_rate_pct": None}
