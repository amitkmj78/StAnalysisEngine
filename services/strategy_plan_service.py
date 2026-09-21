from datetime import datetime, timezone

from .million_plan_service import _future_value

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
    annual_increase_pct: float = 0.0,
) -> dict:
    """
    Expected value if the plan's monthly_contribution (stepping up by
    annual_increase_pct once every 12 months, 0 for plans saved before that
    option existed) had been invested every month since creation at
    annual_return_pct, compared against the user's actual live portfolio
    value. This is a proxy, not a ledger — it assumes the contribution was
    actually made each month, which the app has no way to verify without a
    full contribution log.

    Uses million_plan_service._future_value as its compounding primitive
    (annuity-due: contribute, then grow, each month) instead of a separately
    -compounded starting-capital term -- mathematically identical to the
    previous two-term calculation when annual_increase_pct is 0 (folding a
    constant starting balance into the same contribute-then-grow loop
    distributes no differently than compounding it on its own), so existing
    saved plans' progress numbers are unaffected by this change.
    """
    expected_value = _future_value(starting_capital, monthly_contribution, annual_increase_pct, annual_return_pct, months_elapsed)

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
