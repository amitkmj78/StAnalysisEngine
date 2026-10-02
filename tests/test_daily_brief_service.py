from unittest.mock import patch

from services.daily_brief_service import build_evening_recap


def _performance(rows, total_day_gain=None, total_day_gain_pct=None):
    return {
        "rows": rows,
        "total_day_gain": total_day_gain,
        "total_day_gain_pct": total_day_gain_pct,
    }


def test_build_evening_recap_returns_none_with_no_positions():
    assert build_evening_recap([]) is None


def test_build_evening_recap_returns_none_when_no_row_has_day_gain():
    performance = _performance([{"ticker": "AAA", "day_gain": None, "day_gain_pct": None}])
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=performance):
        result = build_evening_recap([{"ticker": "AAA", "shares": 1.0, "avg_cost": 10.0}])
    assert result is None


def test_build_evening_recap_states_portfolio_and_benchmark_pct():
    rows = [
        {"ticker": "AAA", "day_gain": 50.0, "day_gain_pct": 2.0},
        {"ticker": "BBB", "day_gain": -20.0, "day_gain_pct": -1.0},
    ]
    performance = _performance(rows, total_day_gain=30.0, total_day_gain_pct=1.5)
    positions = [{"ticker": "AAA", "shares": 1.0, "avg_cost": 10.0}, {"ticker": "BBB", "shares": 1.0, "avg_cost": 10.0}]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=performance), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.8
    ):
        result = build_evening_recap(positions)

    assert result is not None
    assert "+1.50%" in result["text_body"]
    assert "+0.80%" in result["text_body"]
    assert "SPY" in result["text_body"]
    assert "portfolio +1.50% today" in result["subject"]


def test_build_evening_recap_ranks_contributors_and_detractors():
    rows = [
        {"ticker": "GAIN_BIG", "day_gain": 100.0, "day_gain_pct": 5.0},
        {"ticker": "GAIN_SMALL", "day_gain": 10.0, "day_gain_pct": 0.5},
        {"ticker": "LOSS_BIG", "day_gain": -80.0, "day_gain_pct": -4.0},
        {"ticker": "LOSS_SMALL", "day_gain": -5.0, "day_gain_pct": -0.2},
    ]
    performance = _performance(rows, total_day_gain=25.0, total_day_gain_pct=1.0)
    positions = [{"ticker": r["ticker"], "shares": 1.0, "avg_cost": 10.0} for r in rows]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=performance), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.5
    ):
        result = build_evening_recap(positions)

    body = result["text_body"]
    contributors_section = body.split("Top contributors:")[1].split("Top detractors:")[0]
    detractors_section = body.split("Top detractors:")[1]

    assert contributors_section.index("GAIN_BIG") < contributors_section.index("GAIN_SMALL")
    assert detractors_section.index("LOSS_BIG") < detractors_section.index("LOSS_SMALL")


def test_build_evening_recap_caps_each_direction_at_three():
    rows = [{"ticker": f"G{i}", "day_gain": float(10 - i), "day_gain_pct": 1.0} for i in range(5)]
    performance = _performance(rows, total_day_gain=40.0, total_day_gain_pct=2.0)
    positions = [{"ticker": r["ticker"], "shares": 1.0, "avg_cost": 10.0} for r in rows]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=performance), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.3
    ):
        result = build_evening_recap(positions)

    contributors_section = result["text_body"].split("Top contributors:")[1]
    assert contributors_section.count("G") == 3
    assert "G3" not in contributors_section and "G4" not in contributors_section
