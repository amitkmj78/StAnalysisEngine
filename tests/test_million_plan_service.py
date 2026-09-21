"""
Strategies page rewrite (ST-1 through ST-6): pure-function coverage for the
new goal-plan solver engine in services/million_plan_service.py. No live
network -- every test uses synthetic, hand-checkable numbers. The existing
required_monthly_investment/project_total_future_value/
build_million_plan_table_from_returns functions are untouched by this
rewrite (still used by the legacy Streamlit page) and are not re-tested
here.
"""

import pytest

from services.million_plan_service import (
    FEASIBILITY_BLOCK_THRESHOLD_PCT,
    FEASIBILITY_WARN_THRESHOLD_PCT,
    TAX_DRAG_PCT_BY_ACCOUNT,
    _future_value,
    check_horizon_conflict,
    compute_goal_plan,
    solve_achievable_amount,
    solve_required_contribution,
    solve_required_return,
    solve_time_to_goal,
)


# ---------------------------------------------------------------------------
# _future_value
# ---------------------------------------------------------------------------


def test_future_value_zero_return_is_capital_plus_contributions():
    fv = _future_value(1000, 100, 0, 0, 12)
    assert fv == pytest.approx(1000 + 100 * 12)


def test_future_value_zero_months_is_just_starting_capital():
    fv = _future_value(5000, 200, 0, 7.0, 0)
    assert fv == pytest.approx(5000)


def test_future_value_matches_textbook_annuity_due_growth():
    # $1000/mo for 12 months at 12%/yr (1%/mo), annuity-due (contribute then
    # grow): FV = 1000 * [((1.01)^12 - 1) / 0.01] * 1.01
    fv = _future_value(0, 1000, 0, 12.0, 12)
    expected = 1000 * (((1.01) ** 12 - 1) / 0.01) * 1.01
    assert fv == pytest.approx(expected, rel=1e-9)


def test_future_value_contribution_steps_up_annually():
    # Two years, contribution starts at 100, steps up 100% after year 1 (so
    # year 2's contribution is 200/mo) -- at 0% return this is just the sum.
    fv = _future_value(0, 100, 100, 0, 24)
    assert fv == pytest.approx(100 * 12 + 200 * 12)


# ---------------------------------------------------------------------------
# Solvers recover a planted value
# ---------------------------------------------------------------------------


def test_solve_required_return_recovers_planted_rate():
    target = _future_value(0, 500, 0, 8.0, 120)
    recovered = solve_required_return(target, 10, 0, 500, 0)
    assert recovered == pytest.approx(8.0, abs=0.01)


def test_solve_required_contribution_recovers_planted_contribution():
    target = _future_value(10_000, 300, 0, 7.0, 60)
    recovered = solve_required_contribution(target, 5, 10_000, 7.0, 0)
    assert recovered == pytest.approx(300, abs=0.5)


def test_solve_time_to_goal_recovers_planted_horizon():
    target = _future_value(0, 400, 0, 6.0, 84)
    recovered = solve_time_to_goal(target, 0, 400, 6.0, 0)
    assert recovered == pytest.approx(7.0, abs=0.05)


def test_solve_achievable_amount_matches_future_value_directly():
    expected = _future_value(2000, 250, 0, 5.0, 36)
    achievable = solve_achievable_amount(3, 2000, 250, 5.0, 0)
    assert achievable == pytest.approx(expected)


def test_solve_required_return_none_when_unreachable_even_at_50pct():
    # $1,000,000 target, 1 year, no capital, $10/mo -- not reachable even
    # at a 50% annual return.
    result = solve_required_return(1_000_000, 1, 0, 10, 0)
    assert result is None


def test_solve_required_contribution_none_when_unreachable_even_at_1m_per_month():
    result = solve_required_contribution(10**12, 1, 0, 5.0, 0)
    assert result is None


def test_solve_time_to_goal_with_inflation_solves_a_moving_target():
    # A today's-dollars target that inflates over the (unknown) horizon --
    # confirm the inflation-aware solve lands noticeably later than solving
    # against the same nominal figure with no inflation, since the target
    # itself is growing over the horizon being searched.
    nominal_years = solve_time_to_goal(100_000, 0, 500, 6.0, 0, inflation_pct=0.0)
    inflating_years = solve_time_to_goal(100_000, 0, 500, 6.0, 0, inflation_pct=3.0)
    assert nominal_years is not None and inflating_years is not None
    assert inflating_years > nominal_years


# ---------------------------------------------------------------------------
# Feasibility gate (ST-3), via compute_goal_plan
# ---------------------------------------------------------------------------


def test_feasibility_ok_below_warn_threshold():
    plan = compute_goal_plan(
        mode="required_return", target_amount=50_000, dollars_mode="future", years=10,
        starting_capital=5000, monthly_contribution=300, annual_contribution_increase_pct=0,
        annual_return_pct=None, inflation_pct=0, account_type="Roth", fund_category="US Large Blend",
    )
    assert plan.feasibility_level == "ok"
    assert plan.gross_return_pct is not None and plan.gross_return_pct < FEASIBILITY_WARN_THRESHOLD_PCT
    assert plan.fixes is None


def test_feasibility_warning_between_thresholds():
    # Given return directly (required_contribution mode) so the gate judges
    # exactly the value supplied, not a solved one.
    plan = compute_goal_plan(
        mode="required_contribution", target_amount=100_000, dollars_mode="future", years=10,
        starting_capital=0, monthly_contribution=None, annual_contribution_increase_pct=0,
        annual_return_pct=13.0, inflation_pct=0, account_type="Taxable", fund_category="All",
    )
    assert plan.feasibility_level == "warning"
    assert plan.gross_return_pct == pytest.approx(13.0)
    assert plan.fixes is None


def test_feasibility_blocked_above_block_threshold_has_three_fixes():
    plan = compute_goal_plan(
        mode="required_contribution", target_amount=100_000, dollars_mode="future", years=10,
        starting_capital=0, monthly_contribution=None, annual_contribution_increase_pct=0,
        annual_return_pct=18.0, inflation_pct=0, account_type="Taxable", fund_category="All",
    )
    assert plan.feasibility_level == "blocked"
    assert plan.fixes is not None
    assert {f["type"] for f in plan.fixes} == {"more_time", "more_contribution", "lower_target"}
    assert plan.fixes[1]["monthly_contribution_needed"] is not None


def test_feasibility_blocked_when_required_return_unreachable():
    # Genuinely infeasible target -- must block, not silently say "ok".
    plan = compute_goal_plan(
        mode="required_return", target_amount=1_000_000, dollars_mode="future", years=3,
        starting_capital=0, monthly_contribution=500, annual_contribution_increase_pct=0,
        annual_return_pct=None, inflation_pct=0, account_type="Taxable", fund_category="All",
    )
    assert plan.feasibility_level == "blocked"
    assert plan.gross_return_pct is None
    assert plan.fixes is not None
    for fix in plan.fixes:
        # None of the three fixes should themselves crash/omit a figure just
        # because the primary solve failed to produce a usable years/return.
        has_figure = any(
            fix.get(k) is not None
            for k in ("years_needed", "monthly_contribution_needed", "achievable_target_future_dollars")
        )
        assert has_figure


def test_feasibility_thresholds_are_the_documented_constants():
    assert FEASIBILITY_WARN_THRESHOLD_PCT == 12.0
    assert FEASIBILITY_BLOCK_THRESHOLD_PCT == 15.0


# ---------------------------------------------------------------------------
# Tax drag (ST-5)
# ---------------------------------------------------------------------------


def test_tax_drag_reduces_net_return_for_taxable_only():
    assert TAX_DRAG_PCT_BY_ACCOUNT["Taxable"] > 0
    assert TAX_DRAG_PCT_BY_ACCOUNT["Traditional"] == 0
    assert TAX_DRAG_PCT_BY_ACCOUNT["Roth"] == 0


def test_required_contribution_higher_for_taxable_than_roth():
    kwargs = dict(mode="required_contribution", target_amount=100_000, dollars_mode="future", years=10,
                   starting_capital=0, monthly_contribution=None, annual_contribution_increase_pct=0,
                   annual_return_pct=8.0, inflation_pct=0, fund_category="All")
    taxable = compute_goal_plan(account_type="Taxable", **kwargs)
    roth = compute_goal_plan(account_type="Roth", **kwargs)
    # Same GROSS return assumed either way, but Taxable's tax drag means
    # less actually compounds, so it needs a bigger contribution to reach
    # the same nominal target.
    assert taxable.solved_value > roth.solved_value


# ---------------------------------------------------------------------------
# Inflation (ST-4)
# ---------------------------------------------------------------------------


def test_inflation_today_future_round_trip():
    plan = compute_goal_plan(
        mode="required_contribution", target_amount=100_000, dollars_mode="today", years=20,
        starting_capital=0, monthly_contribution=None, annual_contribution_increase_pct=0,
        annual_return_pct=6.0, inflation_pct=2.5, account_type="Roth", fund_category="All",
    )
    assert plan.target_today_dollars == pytest.approx(100_000, abs=1)
    assert plan.target_future_dollars > plan.target_today_dollars
    # 20 years at 2.5% inflation should be understated by roughly 60% in
    # nominal terms if inflation were ignored (per the spec's own framing).
    assert plan.target_future_dollars / plan.target_today_dollars == pytest.approx(1.025**20, rel=1e-6)


def test_inflation_future_dollars_mode_uses_target_as_is():
    plan = compute_goal_plan(
        mode="required_contribution", target_amount=200_000, dollars_mode="future", years=20,
        starting_capital=0, monthly_contribution=None, annual_contribution_increase_pct=0,
        annual_return_pct=6.0, inflation_pct=2.5, account_type="Roth", fund_category="All",
    )
    assert plan.target_future_dollars == pytest.approx(200_000, abs=1)
    assert plan.target_today_dollars < plan.target_future_dollars


# ---------------------------------------------------------------------------
# Horizon-based risk tier (ST-6)
# ---------------------------------------------------------------------------


def test_horizon_conflict_short_horizon_equity_category_warns():
    warnings = check_horizon_conflict(2, "US Large Blend", include_stock_picks=True)
    assert len(warnings) == 2  # category warning + stock-picks warning


def test_horizon_conflict_short_horizon_bond_category_no_category_warning():
    warnings = check_horizon_conflict(2, "Bond", include_stock_picks=True)
    assert all("Bond" not in w for w in warnings)
    assert len(warnings) == 1  # just the unconditional stock-picks warning


def test_horizon_conflict_long_horizon_no_warnings():
    warnings = check_horizon_conflict(10, "US Large Blend", include_stock_picks=True)
    assert warnings == []


def test_horizon_conflict_all_category_skipped():
    warnings = check_horizon_conflict(2, "All", include_stock_picks=False)
    assert warnings == []


def test_horizon_conflict_mid_horizon_equity_capped_warning():
    warnings = check_horizon_conflict(5, "US Growth", include_stock_picks=False)
    assert len(warnings) == 1
    assert "conservative mix" in warnings[0]
