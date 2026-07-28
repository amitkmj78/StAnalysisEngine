from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from services.index_fund_service import GOAL_WEIGHTS as FUND_GOAL_WEIGHTS
from services.index_fund_service import LOWER_IS_BETTER as FUND_LOWER_IS_BETTER
from services.index_fund_service import METRIC_LABELS as FUND_METRIC_LABELS
from services.index_fund_service import METRIC_UNITS as FUND_METRIC_UNITS
from services.index_fund_service import rank_index_funds
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

    ranked_funds = rank_index_funds(fund_goal, fund_category)
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


MIN_FUND_PICKS = 10
MIN_STOCK_PICKS = 10


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

    Regardless of top_n, at least MIN_FUND_PICKS funds and MIN_STOCK_PICKS
    stocks are always returned (spread evenly across each goal) so the
    list has real breadth to browse, not just one pick per philosophy.
    """
    picks: list[StrategyPick] = []

    fund_n = max(top_n, -(-MIN_FUND_PICKS // len(FUND_GOAL_WEIGHTS)))
    stock_n = max(top_n, -(-MIN_STOCK_PICKS // len(STOCK_GOAL_WEIGHTS)))

    for fund_goal in FUND_GOAL_WEIGHTS:
        ranked_funds = rank_index_funds(fund_goal, fund_category)
        if ranked_funds.empty:
            continue
        for idx, (_, winner) in enumerate(ranked_funds.head(fund_n).iterrows(), start=1):
            suffix = f" #{idx}" if fund_n > 1 else ""
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
        for idx, (_, winner) in enumerate(ranked_stocks.head(stock_n).iterrows(), start=1):
            suffix = f" #{idx}" if stock_n > 1 else ""
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
