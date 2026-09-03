from datetime import datetime
from unittest.mock import patch

from services.portfolio_performance_service import _compute_position_row, compute_portfolio_performance


def test_day_gain_uses_previous_close():
    pos = {"ticker": "AAA", "shares": 10.0, "avg_cost": 90.0}
    with patch("services.portfolio_performance_service.get_latest_price", return_value=100.0), patch(
        "services.portfolio_performance_service.get_extended_hours_price", return_value=None
    ), patch("services.portfolio_performance_service.get_previous_close", return_value=95.0), patch(
        "services.portfolio_performance_service.price_near_date", return_value=None
    ):
        row = _compute_position_row(pos, datetime.utcnow())

    assert row["day_gain"] == 50.0  # (100 - 95) * 10
    assert row["day_gain_pct"] == 100.0 * (5.0 / 95.0)


def test_day_gain_uses_extended_hours_price_when_available():
    pos = {"ticker": "AAA", "shares": 10.0, "avg_cost": 90.0}
    with patch("services.portfolio_performance_service.get_latest_price", return_value=100.0), patch(
        "services.portfolio_performance_service.get_extended_hours_price",
        return_value={"state": "POST", "price": 103.0, "change_pct": 3.0},
    ), patch("services.portfolio_performance_service.get_previous_close", return_value=95.0), patch(
        "services.portfolio_performance_service.price_near_date", return_value=None
    ):
        row = _compute_position_row(pos, datetime.utcnow())

    # Uses the after-hours price (103), not the regular one (100), for the
    # day-gain calc — matches value_now's own after-hours-aware price.
    assert row["day_gain"] == 80.0  # (103 - 95) * 10


def test_day_gain_is_none_when_previous_close_unavailable():
    pos = {"ticker": "AAA", "shares": 10.0, "avg_cost": 90.0}
    with patch("services.portfolio_performance_service.get_latest_price", return_value=100.0), patch(
        "services.portfolio_performance_service.get_extended_hours_price", return_value=None
    ), patch("services.portfolio_performance_service.get_previous_close", return_value=None), patch(
        "services.portfolio_performance_service.price_near_date", return_value=None
    ):
        row = _compute_position_row(pos, datetime.utcnow())

    assert row["day_gain"] is None
    assert row["day_gain_pct"] is None


def test_compute_portfolio_performance_totals_day_gain_across_positions():
    positions = [
        {"ticker": "AAA", "shares": 10.0, "avg_cost": 90.0},
        {"ticker": "BBB", "shares": 5.0, "avg_cost": 40.0},
    ]
    prices = {"AAA": 100.0, "BBB": 50.0}
    prev_closes = {"AAA": 95.0, "BBB": 48.0}
    with patch(
        "services.portfolio_performance_service.get_latest_price", side_effect=lambda t: prices[t]
    ), patch("services.portfolio_performance_service.get_extended_hours_price", return_value=None), patch(
        "services.portfolio_performance_service.get_previous_close", side_effect=lambda t: prev_closes[t]
    ), patch("services.portfolio_performance_service.price_near_date", return_value=None):
        result = compute_portfolio_performance(positions)

    # AAA: (100-95)*10 = 50, BBB: (50-48)*5 = 10 -> total 60
    assert result["total_day_gain"] == 60.0
    assert result["total_day_gain_pct"] is not None
