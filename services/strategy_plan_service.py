from datetime import datetime, timezone

from .monthly_investing_service import project_future_value_periods

AVG_DAYS_PER_MONTH = 365.25 / 12


def elapsed_months(created_at: datetime, now: datetime | None = None) -> int:
    """Whole months since the plan was created, floored — a plan saved
    an hour ago is 0 months in, not 1, so progress never overstates itself
    on day one."""
    now = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = (now - created_at).total_seconds() / 86400
    return max(0, int(days // AVG_DAYS_PER_MONTH))


def compute_plan_progress(
    starting_capital: float,
    monthly_contribution: float,
    annual_return_pct: float,
    months_elapsed: int,
    current_portfolio_value: float,
) -> dict:
    """
    Expected value if the plan's fixed monthly_contribution had been
    invested every month since creation at annual_return_pct, compared
    against the user's actual live portfolio value. This is a proxy, not
    a ledger — it assumes the contribution was actually made each month,
    which the app has no way to verify without a full contribution log.
    """
    monthly_rate = annual_return_pct / 100 / 12
    contributions_fv = project_future_value_periods(monthly_contribution, months_elapsed, annual_return_pct) or 0.0
    starting_fv = starting_capital * ((1 + monthly_rate) ** months_elapsed)
    expected_value = contributions_fv + starting_fv

    diff = current_portfolio_value - expected_value
    diff_pct = (diff / expected_value * 100.0) if expected_value else None

    return {
        "months_elapsed": months_elapsed,
        "expected_value": round(expected_value, 2),
        "actual_value": round(current_portfolio_value, 2),
        "diff": round(diff, 2),
        "diff_pct": round(diff_pct, 2) if diff_pct is not None else None,
        "on_track": diff >= 0,
    }
