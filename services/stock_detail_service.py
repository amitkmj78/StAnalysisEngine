"""
Pure computation helpers for the stock detail page (docs/stock-analysis-
requirements.html, DET-1..5). No DB/HTTP/yfinance imports here -- every
function takes already-fetched data, so this is fully testable with
synthetic inputs, mirroring this app's established convention of keeping
pure logic separate from the I/O layer that feeds it.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd


def select_peers(ticker: str, universe_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """DET-5: the top_n closest stocks by sector and size -- same GICS
    sector, ranked by absolute market-cap distance from the target
    ticker, ties broken by ticker for determinism. universe_df needs
    "Ticker", "Name", "GICS Sector", "Market Cap ($B)" columns (the same
    shape services.stock_finder_service.get_stock_finder_table already
    produces) -- a sort/filter over data already fetched elsewhere, no
    new fetch layer."""
    ticker = ticker.upper()
    row = universe_df[universe_df["Ticker"] == ticker]
    if row.empty:
        return []
    sector = row.iloc[0].get("GICS Sector")
    market_cap = row.iloc[0].get("Market Cap ($B)")
    if sector is None or pd.isna(market_cap):
        return []

    candidates = universe_df[(universe_df["GICS Sector"] == sector) & (universe_df["Ticker"] != ticker)].copy()
    candidates = candidates.dropna(subset=["Market Cap ($B)"])
    if candidates.empty:
        return []
    candidates["cap_distance"] = (candidates["Market Cap ($B)"] - market_cap).abs()
    candidates = candidates.sort_values(["cap_distance", "Ticker"]).head(top_n)
    return [
        {
            "ticker": r["Ticker"],
            "name": r.get("Name"),
            "market_cap_b": round(float(r["Market Cap ($B)"]), 2),
        }
        for _, r in candidates.iterrows()
    ]


def next_earnings_date(earnings_dates: pd.DataFrame, as_of: Optional[date] = None) -> Optional[dict]:
    """DET-1: the next upcoming earnings date within whatever window
    yfinance's own get_earnings_dates() returned, or None if there isn't
    one in that window (an honest gap, not guessed at) or the ticker has
    no coverage at all."""
    if earnings_dates.empty:
        return None
    index = earnings_dates.index
    if getattr(index, "tz", None) is not None:
        # Same tz-aware-index normalization as portfolio_compare_service.
        # _fetch_close -- comparing a tz-aware index against a tz-naive
        # pd.Timestamp raises TypeError otherwise.
        earnings_dates = earnings_dates.copy()
        earnings_dates.index = index.tz_localize(None)

    as_of_ts = pd.Timestamp(as_of or date.today())
    upcoming = earnings_dates[earnings_dates.index.normalize() >= as_of_ts].sort_index()
    if upcoming.empty:
        return None

    ts = upcoming.index[0]
    eps_estimate = upcoming.iloc[0].get("EPS Estimate")
    return {
        "date": ts.date().isoformat(),
        "eps_estimate": None if eps_estimate is None or pd.isna(eps_estimate) else round(float(eps_estimate), 2),
    }


def recent_dividends(dividends: pd.Series, top_n: int = 4) -> list[dict]:
    """DET-1: the most recent top_n dividend payments, most recent first.
    Empty list for a stock that's never paid one -- not a missing-data
    error, just a fact about that stock."""
    if dividends.empty:
        return []
    tail = dividends.sort_index(ascending=False).head(top_n)
    return [{"date": d.date().isoformat(), "amount": round(float(v), 4)} for d, v in tail.items()]
