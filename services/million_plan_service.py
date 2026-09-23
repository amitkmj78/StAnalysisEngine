from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS
from services.index_fund_service import LOWER_IS_BETTER as FUND_LOWER_IS_BETTER
from services.index_fund_service import METRIC_LABELS as FUND_METRIC_LABELS
from services.index_fund_service import METRIC_UNITS as FUND_METRIC_UNITS
from services.index_fund_service import rank_funds_overall, rank_index_funds
from services.monthly_investing_service import project_future_value
from services.stock_finder_service import GOAL_WEIGHTS as STOCK_GOAL_WEIGHTS
from services.stock_finder_service import LOWER_IS_BETTER as STOCK_LOWER_IS_BETTER
from services.stock_finder_service import METRIC_LABELS as STOCK_METRIC_LABELS
from services.stock_finder_service import METRIC_UNITS as STOCK_METRIC_UNITS
from services.stock_finder_service import rank_stocks


DEFAULT_TARGET_AMOUNT = 1_000_000
DEFAULT_TARGET_YEARS = 5


@dataclass(frozen=True)
class ScoreFactor:
    metric: str
    weight_pct: float
    lower_is_better: bool
    value: float | None = None
    unit: str = ""


@dataclass(frozen=True)
class StrategyPick:
    label: str
    ticker: str
    name: str
    annual_return_pct: float | None
    score: float
    asset_type: str
    score_basis: list[ScoreFactor] = field(default_factory=list)


def _score_basis(
    goal_weights: dict[str, float],
    metric_labels: dict[str, str],
    metric_units: dict[str, str],
    lower_is_better: set[str],
    winner: pd.Series,
) -> list[ScoreFactor]:
    basis = []
    for metric, weight in sorted(goal_weights.items(), key=lambda kv: kv[1], reverse=True):
        raw_value = winner.get(metric)
        value = float(raw_value) if raw_value is not None and pd.notna(raw_value) else None
        basis.append(
            ScoreFactor(
                metric=metric_labels.get(metric, metric),
                weight_pct=round(weight * 100, 1),
                lower_is_better=metric in lower_is_better,
                value=value,
                unit=metric_units.get(metric, ""),
            )
        )
    return basis


def required_monthly_investment(
    target_amount: float,
    years: int,
    annual_return_pct: float,
    starting_capital: float = 0.0,
) -> float:
    monthly_rate = annual_return_pct / 100 / 12
    periods = years * 12

    if monthly_rate == 0:
        return max(0.0, (target_amount - starting_capital) / periods)

    future_value_of_start = starting_capital * ((1 + monthly_rate) ** periods)
    remaining_target = target_amount - future_value_of_start
    if remaining_target <= 0:
        return 0.0

    annuity_factor = (((1 + monthly_rate) ** periods) - 1) / monthly_rate
    return remaining_target / annuity_factor


def project_total_future_value(
    monthly_amount: float,
    years: int,
    annual_return_pct: float | None,
    starting_capital: float = 0.0,
) -> float | None:
    if annual_return_pct is None:
        return None
    projected_from_monthly = project_future_value(monthly_amount, years, annual_return_pct)
    if projected_from_monthly is None:
        return None
    monthly_rate = annual_return_pct / 100 / 12
    starting_capital_future = starting_capital * ((1 + monthly_rate) ** (years * 12))
    return projected_from_monthly + starting_capital_future


def build_million_plan_table(starting_capital: float = 0.0) -> pd.DataFrame:
    return build_million_plan_table_from_returns(
        annual_returns=[6.0, 8.0, 10.0, 12.0, 15.0],
        target_amount=DEFAULT_TARGET_AMOUNT,
        years=DEFAULT_TARGET_YEARS,
        starting_capital=starting_capital,
    )


def build_million_plan_table_from_returns(
    annual_returns: Iterable[float],
    target_amount: float,
    years: int,
    starting_capital: float = 0.0,
) -> pd.DataFrame:
    cleaned_returns = sorted({round(float(value), 2) for value in annual_returns if value is not None})

    rows = []
    for annual_return in cleaned_returns:
        monthly_needed = required_monthly_investment(
            target_amount=target_amount,
            years=years,
            annual_return_pct=annual_return,
            starting_capital=starting_capital,
        )
        projected = project_future_value(monthly_needed, years, annual_return)
        total_contributions = monthly_needed * years * 12
        rows.append(
            {
                "Strategy": f"{annual_return:.1f}% return case",
                "Annual Return %": annual_return,
                "Required Monthly Invest": monthly_needed,
                "Total Contributions": total_contributions,
                "Projected Value": project_total_future_value(
                    monthly_amount=monthly_needed,
                    years=years,
                    annual_return_pct=annual_return,
                    starting_capital=starting_capital,
                ) or 0.0,
            }
        )

    return pd.DataFrame(rows)


def get_million_plan_picks(
    fund_goal: str,
    fund_category: str,
    stock_goal: str,
    stock_universe: str,
    top_n: int = 2,
) -> list[StrategyPick]:
    picks: list[StrategyPick] = []

    ranked_funds, _ = rank_index_funds(fund_goal, fund_category)
    ranked_funds = rank_funds_overall(ranked_funds)
    if not ranked_funds.empty:
        for idx, (_, winner) in enumerate(ranked_funds.head(top_n).iterrows(), start=1):
            picks.append(
                StrategyPick(
                    label=f"Fund Pick {idx}",
                    ticker=str(winner["Ticker"]),
                    name=str(winner["Fund"]),
                    annual_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
                    score=float(winner["Score"]),
                    asset_type="Fund",
                )
            )

    ranked_stocks = rank_stocks(stock_goal, stock_universe)
    if not ranked_stocks.empty:
        for idx, (_, winner) in enumerate(ranked_stocks.head(top_n).iterrows(), start=1):
            picks.append(
                StrategyPick(
                    label=f"Stock Pick {idx}",
                    ticker=str(winner["Ticker"]),
                    name=str(winner["Name"]),
                    annual_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
                    score=float(winner["Score"]),
                    asset_type="Stock",
                )
            )

    return picks


def get_diverse_strategy_picks(
    fund_category: str,
    stock_universe: str,
    top_n: int = 1,
) -> list[StrategyPick]:
    """
    Like get_million_plan_picks, but instead of picks from one hand-picked
    fund goal + one stock goal, pulls picks from *every* available goal
    (all 4 fund philosophies, both stock horizons) — a genuinely diverse
    menu of strategies to compare side by side, not one narrow slice.
    Same underlying ranking functions, just called across the full goal
    space instead of a single selection.

    top_n is honored literally (this used to silently force a higher
    minimum "for breadth," which meant the Strategies page's own "Picks
    per strategy" control had no effect -- a real, reported bug. If more
    breadth is wanted later, it should be a separate, visible control, not
    a silent floor on this one).
    """
    picks: list[StrategyPick] = []

    for fund_goal in FUND_GOAL_WEIGHTS:
        ranked_funds, _ = rank_index_funds(fund_goal, fund_category)
        ranked_funds = rank_funds_overall(ranked_funds)
        if ranked_funds.empty:
            continue
        for idx, (_, winner) in enumerate(ranked_funds.head(top_n).iterrows(), start=1):
            suffix = f" #{idx}" if top_n > 1 else ""
            picks.append(
                StrategyPick(
                    label=f"{fund_goal}{suffix}",
                    ticker=str(winner["Ticker"]),
                    name=str(winner["Fund"]),
                    annual_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
                    score=float(winner["Score"]),
                    asset_type="Fund",
                    score_basis=_score_basis(
                        FUND_GOAL_WEIGHTS[fund_goal], FUND_METRIC_LABELS, FUND_METRIC_UNITS, FUND_LOWER_IS_BETTER, winner
                    ),
                )
            )

    for stock_goal in STOCK_GOAL_WEIGHTS:
        ranked_stocks = rank_stocks(stock_goal, stock_universe)
        if ranked_stocks.empty:
            continue
        for idx, (_, winner) in enumerate(ranked_stocks.head(top_n).iterrows(), start=1):
            suffix = f" #{idx}" if top_n > 1 else ""
            picks.append(
                StrategyPick(
                    label=f"{stock_goal} Stock{suffix}",
                    ticker=str(winner["Ticker"]),
                    name=str(winner["Name"]),
                    annual_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
                    score=float(winner["Score"]),
                    asset_type="Stock",
                    score_basis=_score_basis(
                        STOCK_GOAL_WEIGHTS[stock_goal], STOCK_METRIC_LABELS, STOCK_METRIC_UNITS, STOCK_LOWER_IS_BETTER, winner
                    ),
                )
            )

    return picks


# ---------------------------------------------------------------------------
# Goal-plan solver engine (ST-1 through ST-6). Additive: none of the
# functions/classes above are modified. Standardizes on an annuity-due
# convention (contribute, then grow, each month) throughout -- matching
# project_future_value_periods/project_total_future_value/
# compute_plan_progress above, NOT required_monthly_investment's own
# ordinary-annuity convention (contribute at period end). That mismatch
# between the two pre-existing conventions is real and un-reconciled in
# this file; this new engine doesn't inherit it, it picks one (annuity-due)
# and uses it consistently everywhere below.
# ---------------------------------------------------------------------------

FEASIBILITY_WARN_THRESHOLD_PCT = 12.0
FEASIBILITY_BLOCK_THRESHOLD_PCT = 15.0

# Taxable assumes a modest annual "tax alpha" drag from dividend/turnover
# taxation during accumulation. Traditional/Roth are tax-advantaged during
# accumulation (their tax treatment differs at withdrawal, which this page
# doesn't model) -- 0 drag for both here. Shown to the user, not just coded.
TAX_DRAG_PCT_BY_ACCOUNT = {"Taxable": 0.5, "Traditional": 0.0, "Roth": 0.0}
ACCOUNT_TYPES = list(TAX_DRAG_PCT_BY_ACCOUNT.keys())

SOLVE_MODES = ["required_return", "required_contribution", "time_to_goal", "achievable_amount"]

# Years-to-goal -> permitted risk tier, against this module's own
# (strategies.py's) coarse FUND_CATEGORIES vocabulary -- deliberately not
# index_fund_service's finer-grained category list, since that's not what
# this page's category selector actually offers.
FUND_CATEGORY_RISK_TIER = {
    "All": None,  # ambiguous -- can't judge, no warning
    "Bond": "cash_short",
    "US Large Blend": "growth",
    "US Total Market": "growth",
    "US Growth": "growth",
    "US Small Cap": "growth",
    "International": "growth",
}


def _future_value(
    starting_capital: float,
    monthly_contribution: float,
    annual_increase_pct: float,
    annual_return_pct: float,
    months: int,
) -> float:
    """
    Month-by-month growing-annuity-due simulation: each month, contribute
    then grow. monthly_contribution steps up by annual_increase_pct once
    every 12 months (not compounded monthly). This is the single
    compounding primitive every solver below is built on -- a simulation
    loop rather than four separate closed-form algebraic solutions (one per
    solvable unknown), since keeping four derivations in sync by hand is
    exactly the kind of subtle-bug risk a shared primitive avoids. months is
    bounded low enough by every caller (<=720, i.e. 60 years) that the loop
    cost is negligible even inside a bisection search.

    monthly_rate is the geometric root of annual_return_pct, not
    annual_return_pct/12: dividing by 12 treats the input as a nominal/APR
    rate, which compounds to MORE than the stated annual return once
    applied monthly (a stated 14.5%/12 compounds to ~15.5% effective) --
    silently erasing an assumption like a tax-drag adjustment in the
    process. (1+r)**(1/12)-1 is the rate that actually compounds to
    exactly annual_return_pct over 12 months, matching how "annual return"
    is meant here and everywhere else on this page (the effective rate,
    the way people actually mean "the S&P returns ~10%/year").
    """
    monthly_rate = (1 + annual_return_pct / 100) ** (1 / 12) - 1
    balance = starting_capital
    contribution = monthly_contribution
    for month in range(months):
        if month > 0 and month % 12 == 0:
            contribution *= 1 + annual_increase_pct / 100
        balance = (balance + contribution) * (1 + monthly_rate)
    return balance


def _bisect(fn, lo: float, hi: float, target: float, tol: float = 1e-4, max_iter: int = 60) -> Optional[float]:
    """
    Solves fn(x) == target for x in [lo, hi], where fn is monotonically
    non-decreasing in x. Returns None if target falls outside [fn(lo), fn(hi)]
    (unreachable within the search bounds, in either direction) rather than
    extrapolating a misleading answer.
    """
    fn_lo, fn_hi = fn(lo), fn(hi)
    if target <= fn_lo:
        return lo
    if target > fn_hi:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fn_mid = fn(mid)
        if abs(fn_mid - target) <= tol * max(1.0, abs(target)):
            return mid
        if fn_mid < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_required_return(
    target: float, years: float, starting_capital: float, monthly_contribution: float, annual_increase_pct: float
) -> Optional[float]:
    """Annual return % (net of any tax drag -- caller adds drag back for
    display) needed to reach `target`. None if unreachable even at 50%, or
    if the target is already met at -20% (contribution/capital alone carry
    it, no return is "needed")."""
    months = round(years * 12)
    return _bisect(lambda r: _future_value(starting_capital, monthly_contribution, annual_increase_pct, r, months), -20.0, 50.0, target)


def solve_required_contribution(
    target: float, years: float, starting_capital: float, annual_return_pct: float, annual_increase_pct: float
) -> Optional[float]:
    """Monthly contribution needed to reach `target`. None if unreachable
    even at $1,000,000/month (target/timeframe combination is not viable
    regardless of contribution size)."""
    months = round(years * 12)
    return _bisect(
        lambda c: _future_value(starting_capital, c, annual_increase_pct, annual_return_pct, months), 0.0, 1_000_000.0, target
    )


def solve_time_to_goal(
    target: float,
    starting_capital: float,
    monthly_contribution: float,
    annual_return_pct: float,
    annual_increase_pct: float,
    max_years: float = 60.0,
    inflation_pct: float = 0.0,
) -> Optional[float]:
    """
    Years needed to reach `target`. When inflation_pct > 0, `target` is
    treated as a TODAY'S-dollars figure that itself grows with inflation
    over the (unknown) horizon being solved for -- i.e. this solves for the
    first year where the nominal ending balance matches what `target` costs
    in THAT year's dollars, not a fixed nominal number (a real chicken-and-
    egg problem: converting a today's-dollars target to nominal dollars
    normally needs the number of years, which is exactly what's unknown
    here). Pass the default inflation_pct=0 to solve against an already-
    resolved fixed nominal target instead (degenerates to the fixed-target
    case, since (1+0)**years == 1). None if unreachable within max_years.
    """
    def gap(years_elapsed: float) -> float:
        fv = _future_value(starting_capital, monthly_contribution, annual_increase_pct, annual_return_pct, round(years_elapsed * 12))
        inflated_target = target * (1 + inflation_pct / 100) ** years_elapsed
        return fv - inflated_target

    lo, hi = 0.0, max_years
    if gap(lo) >= 0:
        return lo
    if gap(hi) < 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if gap(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def solve_achievable_amount(
    years: float, starting_capital: float, monthly_contribution: float, annual_return_pct: float, annual_increase_pct: float
) -> float:
    """No solving needed -- this mode's output IS the direct simulation."""
    return _future_value(starting_capital, monthly_contribution, annual_increase_pct, annual_return_pct, round(years * 12))


def check_horizon_conflict(years: float, fund_category: str, include_stock_picks: bool) -> list[str]:
    """
    Years-to-goal constrains what risk level is appropriate, independent of
    the arithmetic (ST-6). Returns human-readable warning strings (empty if
    no conflict). "All" or an unmapped category is skipped -- can't judge a
    category the page doesn't have a tier for.
    """
    warnings: list[str] = []
    tier = FUND_CATEGORY_RISK_TIER.get(fund_category)

    if years < 3:
        if tier == "growth":
            warnings.append(
                f'A {years:g}-year horizon calls for cash and short-duration bonds -- "{fund_category}" is an equity '
                "category and carries meaningfully more risk than this timeframe usually allows for."
            )
        if include_stock_picks:
            warnings.append(
                f"Individual stock picks are equity risk, which is generally not appropriate for a {years:g}-year horizon."
            )
    elif years < 7:
        if tier == "growth":
            warnings.append(
                f'A {years:g}-year horizon calls for a conservative mix with equity capped -- "{fund_category}" is a '
                "full growth category; consider a more conservative source or capping how much of the plan it drives."
            )
        if include_stock_picks:
            warnings.append(
                f"Individual stock picks are equity risk; at a {years:g}-year horizon, consider limiting how much of "
                "the plan relies on them."
            )

    return warnings


# Shown as a companion table whenever the plan is in the warning/blocked
# feasibility band, so the message isn't just "that's unrealistic" -- it's
# "here's what it actually costs at return assumptions people would
# consider reasonable."
RETURN_ASSUMPTION_TABLE_PCTS = [6.0, 8.0, 10.0, 12.0]


def _return_assumption_table(
    target_future: float, years: float, starting_capital: float, annual_increase_pct: float, tax_drag_pct: float
) -> list[dict]:
    rows = []
    for gross_pct in RETURN_ASSUMPTION_TABLE_PCTS:
        net_pct = gross_pct - tax_drag_pct
        contribution = solve_required_contribution(target_future, years, starting_capital, net_pct, annual_increase_pct)
        rows.append(
            {
                "annual_return_pct": gross_pct,
                "monthly_contribution_needed": round(contribution, 2) if contribution is not None else None,
            }
        )
    return rows


@dataclass(frozen=True)
class GoalPlanResult:
    mode: str
    target_today_dollars: float
    target_future_dollars: float
    years: float
    starting_capital: float
    monthly_contribution: float
    annual_contribution_increase_pct: float
    account_type: str
    tax_drag_pct: float
    inflation_pct: float
    solved_value: Optional[float]
    solved_field_label: str
    gross_return_pct: Optional[float]
    net_return_pct: Optional[float]
    feasibility_level: str
    feasibility_message: Optional[str]
    fixes: Optional[list[dict]]
    return_assumption_table: Optional[list[dict]]
    horizon_warnings: list[str]


_SOLVED_FIELD_LABELS = {
    "required_return": "Annual return needed",
    "required_contribution": "Monthly contribution needed",
    "time_to_goal": "Years needed",
    "achievable_amount": "Ending balance",
}


def compute_goal_plan(
    mode: str,
    target_amount: Optional[float],
    dollars_mode: str,
    years: Optional[float],
    starting_capital: float,
    monthly_contribution: Optional[float],
    annual_contribution_increase_pct: float,
    annual_return_pct: Optional[float],
    inflation_pct: float,
    account_type: str,
    fund_category: str,
    include_stock_picks: bool = True,
) -> GoalPlanResult:
    """
    Orchestrator for the Strategies page's plan: resolves the today's/future
    -dollars target, resolves tax drag from account_type, dispatches to the
    right solver for `mode`, judges the resulting return against the
    feasibility gate (ST-3) -- computing the three fixes when blocked -- and
    runs the horizon-conflict check (ST-6). Router-level param validation
    (required-field-per-mode, allowed enum values) happens in the caller;
    this function assumes its inputs are already validated for the given
    mode and does no HTTP-shaped error handling.
    """
    tax_drag_pct = TAX_DRAG_PCT_BY_ACCOUNT[account_type]

    solved_value: Optional[float] = None
    gross_return_pct: Optional[float] = annual_return_pct
    net_return_pct: Optional[float] = None if annual_return_pct is None else annual_return_pct - tax_drag_pct

    if mode == "time_to_goal":
        # years is the thing being solved -- can't convert a today's-dollars
        # target to nominal dollars before knowing it (inflating over an
        # unknown horizon), so this mode alone defers the today/future
        # conversion until after solving, and solve_time_to_goal itself
        # handles the today's-dollars case (see its own docstring).
        if dollars_mode == "today" and target_amount is not None:
            solved_value = solve_time_to_goal(
                target_amount, starting_capital, monthly_contribution, net_return_pct, annual_contribution_increase_pct, inflation_pct=inflation_pct
            )
        else:
            solved_value = solve_time_to_goal(
                target_amount or 0.0, starting_capital, monthly_contribution, net_return_pct, annual_contribution_increase_pct
            )
        years = solved_value if solved_value is not None else years
        inflation_factor = (1 + inflation_pct / 100) ** (years or 0)
        if target_amount is not None:
            if dollars_mode == "today":
                target_today = target_amount
                target_future = target_amount * inflation_factor
            else:
                target_future = target_amount
                target_today = target_amount / inflation_factor if inflation_factor else target_amount
        else:
            target_today = target_future = 0.0
    else:
        inflation_factor = (1 + inflation_pct / 100) ** (years or 0)

        # Resolve both dollar figures for the "given" target, where one
        # exists. In achievable_amount mode there is no target yet -- both
        # stay at 0.0 here and get overwritten below once the solved ending
        # balance is known.
        if target_amount is not None:
            if dollars_mode == "today":
                target_today = target_amount
                target_future = target_amount * inflation_factor
            else:
                target_future = target_amount
                target_today = target_amount / inflation_factor if inflation_factor else target_amount
        else:
            target_today = target_future = 0.0

        if mode == "required_return":
            net_solved = solve_required_return(target_future, years, starting_capital, monthly_contribution, annual_contribution_increase_pct)
            net_return_pct = net_solved
            gross_return_pct = None if net_solved is None else net_solved + tax_drag_pct
            solved_value = gross_return_pct
        elif mode == "required_contribution":
            solved_value = solve_required_contribution(target_future, years, starting_capital, net_return_pct, annual_contribution_increase_pct)
        elif mode == "achievable_amount":
            solved_value = solve_achievable_amount(years, starting_capital, monthly_contribution, net_return_pct, annual_contribution_increase_pct)
            target_future = solved_value
            target_today = solved_value / inflation_factor if inflation_factor else solved_value

    # Feasibility gate: judged on the GROSS (pre-tax-drag) return, since
    # that's the market-return assumption actually being made. `None` means
    # solve_required_return couldn't reach the target even at 50% -- that's
    # not "ok", it's the single worst case this gate exists to catch, so it
    # blocks exactly like an explicit >15% figure does.
    # years/monthly_contribution can themselves be unresolved by this point
    # (e.g. mode="time_to_goal" failed to converge even at 60 years, or
    # mode="required_contribution" never had a given contribution to begin
    # with) -- these safe fallbacks are shared by the "blocked" three-fixes
    # computation and the warning/blocked return-assumption table below,
    # rather than crashing either on a None.
    years_for_fixes = years if years else 30.0
    contribution_for_fixes = (
        monthly_contribution
        if monthly_contribution is not None
        else (solved_value if mode == "required_contribution" and solved_value is not None else 0.0)
    )

    if gross_return_pct is None or gross_return_pct > FEASIBILITY_BLOCK_THRESHOLD_PCT:
        feasibility_level = "blocked"
        feasibility_message = (
            "This target isn't reachable at any realistic return -- here's what would have to change instead:"
            if gross_return_pct is None
            else (
                f"This plan needs a {gross_return_pct:.1f}% annual return, which is not a realistic assumption to plan "
                "around. Here's what would have to change instead:"
            )
        )
        fix_years = solve_time_to_goal(
            target_future, starting_capital, contribution_for_fixes, FEASIBILITY_WARN_THRESHOLD_PCT - tax_drag_pct, annual_contribution_increase_pct
        )
        fix_contribution = solve_required_contribution(
            target_future, years_for_fixes, starting_capital, FEASIBILITY_WARN_THRESHOLD_PCT - tax_drag_pct, annual_contribution_increase_pct
        )
        fix_target_future = solve_achievable_amount(
            years_for_fixes, starting_capital, contribution_for_fixes, FEASIBILITY_WARN_THRESHOLD_PCT - tax_drag_pct, annual_contribution_increase_pct
        )
        fixes = [
            {
                "type": "more_time",
                "label": "Give it more time",
                "years_needed": round(fix_years, 1) if fix_years is not None else None,
            },
            {
                "type": "more_contribution",
                "label": "Contribute more each month",
                "monthly_contribution_needed": round(fix_contribution, 2) if fix_contribution is not None else None,
            },
            {
                "type": "lower_target",
                "label": "Lower the target",
                "achievable_target_future_dollars": round(fix_target_future, 2),
                "achievable_target_today_dollars": round(fix_target_future / inflation_factor, 2) if inflation_factor else round(fix_target_future, 2),
            },
        ]
        return_assumption_table = _return_assumption_table(
            target_future, years_for_fixes, starting_capital, annual_contribution_increase_pct, tax_drag_pct
        )
    elif gross_return_pct > FEASIBILITY_WARN_THRESHOLD_PCT:
        feasibility_level = "warning"
        feasibility_message = (
            f"This plan needs a {gross_return_pct:.1f}% annual return. That's above what's typically assumed "
            "achievable at this contribution level -- treat it as a stretch goal, not a plan."
        )
        fixes = None
        return_assumption_table = _return_assumption_table(
            target_future, years_for_fixes, starting_capital, annual_contribution_increase_pct, tax_drag_pct
        )
    else:
        feasibility_level = "ok"
        feasibility_message = None
        fixes = None
        return_assumption_table = None

    horizon_warnings = check_horizon_conflict(years or 0, fund_category, include_stock_picks) if years else []

    return GoalPlanResult(
        mode=mode,
        target_today_dollars=round(target_today, 2),
        target_future_dollars=round(target_future, 2),
        years=round(years, 2) if years is not None else 0.0,
        starting_capital=starting_capital,
        monthly_contribution=round(monthly_contribution, 2) if monthly_contribution is not None else (
            round(solved_value, 2) if mode == "required_contribution" and solved_value is not None else 0.0
        ),
        annual_contribution_increase_pct=annual_contribution_increase_pct,
        account_type=account_type,
        tax_drag_pct=tax_drag_pct,
        inflation_pct=inflation_pct,
        solved_value=round(solved_value, 4) if solved_value is not None else None,
        solved_field_label=_SOLVED_FIELD_LABELS[mode],
        gross_return_pct=round(gross_return_pct, 4) if gross_return_pct is not None else None,
        net_return_pct=round(net_return_pct, 4) if net_return_pct is not None else None,
        feasibility_level=feasibility_level,
        feasibility_message=feasibility_message,
        fixes=fixes,
        return_assumption_table=return_assumption_table,
        horizon_warnings=horizon_warnings,
    )
