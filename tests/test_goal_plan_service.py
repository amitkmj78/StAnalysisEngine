import pytest

from services.goal_plan_service import build_signal_weighted_allocation, solve_goal_plan


def test_allocation_tilts_toward_stronger_signal():
    rows = [
        {"ticker": "AAA", "expected_return_pct": 5.0},
        {"ticker": "BBB", "expected_return_pct": -2.0},
    ]
    result = build_signal_weighted_allocation(rows)
    by_ticker = {r["ticker"]: r["weight_pct"] for r in result}
    assert by_ticker["AAA"] > by_ticker["BBB"]
    assert pytest.approx(sum(by_ticker.values()), abs=0.01) == 100.0


def test_allocation_keeps_a_floor_for_the_weakest_signal():
    rows = [
        {"ticker": "AAA", "expected_return_pct": 5.0},
        {"ticker": "BBB", "expected_return_pct": -50.0},
    ]
    result = build_signal_weighted_allocation(rows)
    weakest = next(r for r in result if r["ticker"] == "BBB")
    assert weakest["weight_pct"] > 0


def test_allocation_falls_back_to_equal_weight_with_no_signal_data():
    rows = [
        {"ticker": "AAA", "expected_return_pct": None},
        {"ticker": "BBB", "expected_return_pct": None},
        {"ticker": "CCC", "expected_return_pct": None},
    ]
    result = build_signal_weighted_allocation(rows)
    for r in result:
        assert r["weight_pct"] == pytest.approx(100.0 / 3, abs=0.01)


def test_allocation_unscored_ticker_gets_only_the_floor():
    rows = [
        {"ticker": "AAA", "expected_return_pct": 5.0},
        {"ticker": "BBB", "expected_return_pct": None},
    ]
    result = build_signal_weighted_allocation(rows)
    by_ticker = {r["ticker"]: r["weight_pct"] for r in result}
    assert by_ticker["BBB"] < by_ticker["AAA"]
    assert pytest.approx(sum(by_ticker.values()), abs=0.01) == 100.0


def test_solve_goal_plan_zero_return_splits_shortfall_evenly_across_months():
    plan = solve_goal_plan(
        current_value=0.0,
        target_amount=1200.0,
        months=12,
        current_holdings_annualized_return_pct=0.0,
        contribution_annualized_return_pct=0.0,
    )
    assert plan["required_monthly_contribution"] == pytest.approx(100.0)


def test_solve_goal_plan_already_on_track_needs_no_contribution():
    plan = solve_goal_plan(
        current_value=1_000_000.0,
        target_amount=10_000.0,
        months=12,
        current_holdings_annualized_return_pct=5.0,
        contribution_annualized_return_pct=5.0,
    )
    assert plan["required_monthly_contribution"] == 0.0


def test_solve_goal_plan_positive_return_needs_less_than_zero_return_case():
    zero_return_plan = solve_goal_plan(
        current_value=0.0,
        target_amount=120_000.0,
        months=120,
        current_holdings_annualized_return_pct=0.0,
        contribution_annualized_return_pct=0.0,
    )
    growth_plan = solve_goal_plan(
        current_value=0.0,
        target_amount=120_000.0,
        months=120,
        current_holdings_annualized_return_pct=0.0,
        contribution_annualized_return_pct=8.0,
    )
    assert growth_plan["required_monthly_contribution"] < zero_return_plan["required_monthly_contribution"]


def test_solve_goal_plan_reports_gap_when_monthly_amount_given():
    plan = solve_goal_plan(
        current_value=0.0,
        target_amount=1_000_000.0,
        months=12,
        current_holdings_annualized_return_pct=0.0,
        contribution_annualized_return_pct=0.0,
        monthly_amount=100.0,
    )
    assert plan["projected_value_with_given_contribution"] == pytest.approx(1200.0)
    assert plan["gap_vs_target"] == pytest.approx(1200.0 - 1_000_000.0)
