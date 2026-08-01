from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from services.cache_utils import ttl_cache
from services.index_fund_service import rank_index_funds
from services.stock_finder_service import rank_stocks


@dataclass(frozen=True)
class Recommendation:
    ticker: str
    name: str
    score: float
    asset_type: str
    expected_return_pct: float | None


def get_best_monthly_pick(asset_type: str, goal: str, category_or_universe: str) -> Recommendation | None:
    if asset_type == "Fund":
        ranked = rank_index_funds(goal, category_or_universe)
        if ranked.empty:
            return None
        winner = ranked.iloc[0]
        return Recommendation(
            ticker=str(winner["Ticker"]),
            name=str(winner["Fund"]),
            score=float(winner["Score"]),
            asset_type="Fund",
            expected_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
        )

    ranked = rank_stocks(goal, category_or_universe)
    if ranked.empty:
        return None
    winner = ranked.iloc[0]
    return Recommendation(
        ticker=str(winner["Ticker"]),
        name=str(winner["Name"]),
        score=float(winner["Score"]),
        asset_type="Stock",
        expected_return_pct=float(winner["3Y Annualized %"]) if pd.notna(winner["3Y Annualized %"]) else None,
    )


@ttl_cache(maxsize=128, ttl_seconds=3600)
def get_price_history(ticker: str, years: int) -> pd.DataFrame:
    period = f"{max(years, 1)}y"
    return yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True).dropna()


def simulate_monthly_plan(
    ticker: str,
    monthly_amount: float,
    years: int,
) -> tuple[pd.DataFrame, dict]:
    hist = get_price_history(ticker, years)
    if hist.empty:
        return pd.DataFrame(), {}

    monthly_close = hist["Close"].resample("ME").last().dropna()
    if monthly_close.empty:
        return pd.DataFrame(), {}

    months_needed = min(years * 12, len(monthly_close))
    monthly_close = monthly_close.tail(months_needed)

    invested = 0.0
    shares = 0.0
    rows: list[dict] = []

    for date, price in monthly_close.items():
        bought_shares = monthly_amount / float(price)
        shares += bought_shares
        invested += monthly_amount
        portfolio_value = shares * float(price)
        rows.append(
            {
                "Date": date,
                "Monthly Contribution": monthly_amount,
                "Price": float(price),
                "Shares Bought": bought_shares,
                "Total Invested": invested,
                "Portfolio Value": portfolio_value,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result, {}

    ending_value = float(result["Portfolio Value"].iloc[-1])
    total_invested = float(result["Total Invested"].iloc[-1])
    gain = ending_value - total_invested
    gain_pct = (gain / total_invested * 100) if total_invested else 0.0

    summary = {
        "months": int(len(result)),
        "total_invested": total_invested,
        "ending_value": ending_value,
        "gain": gain,
        "gain_pct": gain_pct,
        "latest_price": float(result["Price"].iloc[-1]),
    }
    return result, summary


def project_future_value(monthly_amount: float, years: int, annual_return_pct: float | None) -> float | None:
    if annual_return_pct is None:
        return None
    monthly_rate = annual_return_pct / 100 / 12
    periods = years * 12
    future_value = 0.0
    for _ in range(periods):
        future_value = (future_value + monthly_amount) * (1 + monthly_rate)
    return future_value
