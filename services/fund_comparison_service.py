from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import Optional

import pandas as pd

from .data_service import get_stock_data
from .index_fund_service import MAX_PARALLEL_FETCHES, get_index_fund_table, rank_index_funds


def get_top_fund(goal: str = "Balanced Core", category: str = "All") -> Optional[dict]:
    """Whichever fund currently ranks #1 for the given goal/category — the
    same ranking already shown on /index-fund and /strategies, reused here
    as the benchmark for "top-performing fund"."""
    df = rank_index_funds(goal, category)
    if df.empty:
        return None
    winner = df.iloc[0]
    return {"ticker": str(winner["Ticker"]), "name": str(winner["Fund"])}


def price_near_date(ticker: str, when: datetime, period: str = "2y") -> Optional[float]:
    """
    Closing price at or shortly after `when` — used to mark a benchmark's
    starting point for "return since you saved this prediction." Falls back
    to the most recent close if `when` is more recent than available history.

    `period` defaults to "2y" (cheap, covers every existing caller's use
    case — recent prediction/portfolio dates). Pass "max" for anything that
    might reach back further than that, e.g. a fund's inception date.
    """
    data = get_stock_data(ticker, period)
    if data.empty:
        return None
    when_naive = when.replace(tzinfo=None) if when.tzinfo else when
    for ts, row in data.iterrows():
        ts_naive = ts.to_pydatetime().replace(tzinfo=None)
        if ts_naive >= when_naive:
            return float(row["Close"])
    return float(data["Close"].iloc[-1])


def rank_funds_by_inception(min_years: int, category: str = "All") -> pd.DataFrame:
    """
    Funds with at least `min_years` of real trading history since their
    inception date, ranked by real since-inception % return (inception
    price vs. current price, via price_near_date's "max"-period lookup) —
    a proven, long-run track record view, distinct from the 1Y/3Y-weighted
    Score the Fund Screener's goal-based ranking uses. Funds missing an
    inception date (not every fund discloses it) are excluded rather than
    treated as ineligible-or-eligible by guesswork.
    """
    df = get_index_fund_table()
    if df.empty or "Inception Date" not in df.columns:
        return pd.DataFrame()

    if category != "All":
        df = df[df["Category"] == category]
    if df.empty:
        return df

    today = date.today()

    def _years_since(inception_str):
        if not inception_str:
            return None
        try:
            inception_date = datetime.strptime(inception_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return None
        return (today - inception_date).days / 365.25

    df = df.copy()
    df["Years Since Inception"] = df["Inception Date"].apply(_years_since)
    eligible = df[df["Years Since Inception"].notna() & (df["Years Since Inception"] >= min_years)].copy()
    if eligible.empty:
        return eligible

    def _return_since_inception(row) -> Optional[float]:
        inception_dt = datetime.strptime(row["Inception Date"], "%Y-%m-%d")
        price_then = price_near_date(row["Ticker"], inception_dt, period="max")
        price_now = row["Price"]
        if not price_then or not price_now:
            return None
        return round((price_now - price_then) / price_then * 100, 2)

    rows = [r for _, r in eligible.iterrows()]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        returns = list(executor.map(_return_since_inception, rows))
    eligible["Since Inception Return %"] = returns
    eligible["Years Since Inception"] = eligible["Years Since Inception"].round(1)

    return (
        eligible.sort_values("Since Inception Return %", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
