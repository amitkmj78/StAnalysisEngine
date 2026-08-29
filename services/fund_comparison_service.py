from datetime import datetime
from typing import Optional

from .data_service import get_stock_data
from .index_fund_service import rank_index_funds


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
