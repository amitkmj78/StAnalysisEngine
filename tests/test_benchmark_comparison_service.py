from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

import pytest

from services.benchmark_comparison_service import (
    UNDERPERFORM_THRESHOLD_PCT,
    compute_benchmark_comparison,
)


def _mock_performance(total_gain_vs_cost_pct, rows=None):
    return {
        "lookback_days": 30,
        "rows": rows or [],
        "total_value_now": 1000.0,
        "total_value_30d_ago": 900.0,
        "value_diff": 100.0,
        "value_diff_pct": 11.1,
        "total_cost_basis": 800.0,
        "total_gain_vs_cost": 200.0,
        "total_gain_vs_cost_pct": total_gain_vs_cost_pct,
        "total_day_gain": 10.0,
        "total_day_gain_pct": 1.0,
    }


@contextmanager
def _patched(performance, price_then=100.0, price_now=110.0, prev_close=108.0):
    with patch(
        "services.benchmark_comparison_service.compute_portfolio_performance", return_value=performance
    ), patch("services.benchmark_comparison_service.price_near_date", return_value=price_then), patch(
        "services.benchmark_comparison_service.get_effective_price", return_value=price_now
    ), patch("services.benchmark_comparison_service.get_previous_close", return_value=prev_close):
        yield


def test_outperforming_when_portfolio_beats_benchmark():
    with _patched(_mock_performance(25.0)):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    assert result["portfolio_return_pct"] == 25.0
    assert result["benchmark_return_pct"] == pytest.approx(10.0)
    assert result["gap_pct"] == pytest.approx(15.0)
    assert result["underperforming"] is False
    assert result["suggestion"] is None


def test_underperforming_names_worst_positions():
    rows = [
        {"ticker": "WINNER", "gain_vs_cost_pct": 20.0, "gain_vs_cost": 100.0, "value_now": 600.0},
        {"ticker": "LOSER1", "gain_vs_cost_pct": -30.0, "gain_vs_cost": -150.0, "value_now": 200.0},
        {"ticker": "LOSER2", "gain_vs_cost_pct": -10.0, "gain_vs_cost": -20.0, "value_now": 180.0},
    ]
    with _patched(_mock_performance(-2.0, rows)):
        result = compute_benchmark_comparison([{"ticker": "X", "shares": 1.0}], datetime(2026, 1, 1))

    # portfolio -2%, benchmark +10% -> gap -12, well past the threshold
    assert result["gap_pct"] == pytest.approx(-12.0)
    assert result["underperforming"] is True
    assert [p["ticker"] for p in result["worst_positions"]] == ["LOSER1", "LOSER2"]
    assert "LOSER1" in result["suggestion"]
    assert "WINNER" not in result["suggestion"]


def test_underperforming_with_no_losing_positions_gets_relative_strength_message():
    rows = [{"ticker": "AAA", "gain_vs_cost_pct": 3.0, "gain_vs_cost": 30.0, "value_now": 500.0}]
    with _patched(_mock_performance(3.0, rows)):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    assert result["underperforming"] is True
    assert result["worst_positions"] == []
    assert "relative strength" in result["suggestion"]


def test_small_gap_within_threshold_does_not_flag():
    with _patched(_mock_performance(9.0)):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    # gap is exactly -1.0, inside the +/-UNDERPERFORM_THRESHOLD_PCT buffer
    assert result["gap_pct"] == pytest.approx(-1.0)
    assert -UNDERPERFORM_THRESHOLD_PCT < result["gap_pct"]
    assert result["underperforming"] is False


def test_missing_benchmark_price_returns_none_without_crashing():
    with _patched(_mock_performance(5.0), price_then=None):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    assert result["benchmark_return_pct"] is None
    assert result["gap_pct"] is None
    assert result["underperforming"] is False


def test_benchmark_today_pct_uses_previous_close():
    # price_now=110, prev_close=108 -> +1.85% today, independent of the
    # since-inception price_then=100 used for benchmark_return_pct.
    with _patched(_mock_performance(5.0), price_then=100.0, price_now=110.0, prev_close=108.0):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    assert result["benchmark_today_pct"] == pytest.approx((110.0 / 108.0 - 1.0) * 100.0)


def test_benchmark_today_pct_none_when_previous_close_unavailable():
    with _patched(_mock_performance(5.0), prev_close=None):
        result = compute_benchmark_comparison([{"ticker": "AAA", "shares": 1.0}], datetime(2026, 1, 1))

    assert result["benchmark_today_pct"] is None
