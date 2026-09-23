"""
compute_plan_progress was switched from a two-term calculation (contributions
via project_future_value_periods + starting capital compounded separately)
to services.million_plan_service._future_value (a single growing-annuity-due
loop), to support ST-1's contribution step-up. At the time, this was
numerically identical to the old calculation for annual_increase_pct=0 --
_future_value's monthly rate was later corrected from a nominal/APR
(annual/12) convention to the geometric monthly rate that actually
compounds to the stated annual return (a separate, reported bug fix), so
these tests now pin down that _future_value's own convention, not a literal
byte-for-byte match to the pre-fix calculation.
"""

import pytest

from services.strategy_plan_service import compute_plan_progress, elapsed_months
from datetime import datetime, timedelta, timezone


def test_progress_matches_effective_annual_rate_convention_at_zero_increase():
    # Reference calculation using _future_value's own (corrected) monthly
    # rate: the geometric root of the annual rate, not annual/12.
    monthly_rate = (1 + 7.0 / 100) ** (1 / 12) - 1
    months = 18
    expected_contributions_fv = 0.0
    for _ in range(months):
        expected_contributions_fv = (expected_contributions_fv + 400) * (1 + monthly_rate)
    expected_starting_fv = 10_000 * ((1 + monthly_rate) ** months)
    expected = expected_contributions_fv + expected_starting_fv

    result = compute_plan_progress(
        starting_capital=10_000, monthly_contribution=400, annual_return_pct=7.0,
        months_elapsed=months, current_portfolio_value=0,
    )
    # expected_value is rounded to cents by compute_plan_progress -- compare
    # with a cent-scale tolerance, not a relative one.
    assert result["expected_value"] == pytest.approx(expected, abs=0.01)


def test_progress_on_track_when_actual_meets_expected():
    result = compute_plan_progress(
        starting_capital=0, monthly_contribution=100, annual_return_pct=6.0,
        months_elapsed=12, current_portfolio_value=10**9,
    )
    assert result["on_track"] is True
    assert result["diff"] > 0


def test_progress_behind_pace_when_actual_below_expected():
    result = compute_plan_progress(
        starting_capital=0, monthly_contribution=100, annual_return_pct=6.0,
        months_elapsed=12, current_portfolio_value=0,
    )
    assert result["on_track"] is False
    assert result["diff"] < 0


def test_progress_with_contribution_step_up_exceeds_flat_contribution():
    flat = compute_plan_progress(
        starting_capital=0, monthly_contribution=500, annual_return_pct=5.0,
        months_elapsed=36, current_portfolio_value=0, annual_increase_pct=0,
    )
    stepped = compute_plan_progress(
        starting_capital=0, monthly_contribution=500, annual_return_pct=5.0,
        months_elapsed=36, current_portfolio_value=0, annual_increase_pct=10,
    )
    assert stepped["expected_value"] > flat["expected_value"]


def test_elapsed_months_floors_and_never_negative():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert elapsed_months(now - timedelta(days=40), now=now) == 1
    assert elapsed_months(now - timedelta(hours=1), now=now) == 0
    assert elapsed_months(now + timedelta(days=1), now=now) == 0
