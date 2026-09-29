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

# DET-3: a single evaluation horizon per score, picked from within SCR-1's
# own stated ranges (short = 10-90 days, long = 1-3 years) rather than
# re-deriving one per row -- 10 trading days is the earliest point a
# short-term call is honestly checkable; 252 (~1 trading year) is the low
# end of the long-term range. Both are trading days, not calendar days,
# matching every other horizon-evaluation in this app (e.g.
# signal_publication_service.evaluate_signal_outcomes_for_date).
DET3_SHORT_HORIZON_DAYS = 10
DET3_LONG_HORIZON_DAYS = 252


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


def evaluate_signal_outcome(as_of_date: date, signal: str, closes: pd.Series, horizon_days: int) -> Optional[dict]:
    """DET-3: hit/miss for one historical signal against this ticker's own
    close-price series (date/Timestamp-indexed, sorted ascending) -- entry
    at the first close on or after as_of_date, exit `horizon_days` trading
    days later. Returns None when the horizon hasn't elapsed yet in the
    series (an honest "not matured yet" gap, never guessed at) or
    as_of_date is past the end of the series entirely.

    Buy is a hit if the stock actually rose; Trim is a hit if it didn't;
    Hold makes no directional call, so it gets no hit/miss verdict -- just
    the realized return, same "what happened" fact every other signal
    gets."""
    if closes.empty:
        return None
    index = closes.index
    if getattr(index, "tz", None) is not None:
        # Same tz-aware-index normalization as next_earnings_date above --
        # yfinance returns a tz-aware (America/New_York) index; comparing
        # it against a tz-naive pd.Timestamp raises TypeError otherwise.
        closes = closes.copy()
        closes.index = index.tz_localize(None)
        index = closes.index
    on_or_after = index[index >= pd.Timestamp(as_of_date)]
    if len(on_or_after) == 0:
        return None
    entry_idx = index.get_loc(on_or_after[0])

    exit_idx = entry_idx + horizon_days
    if exit_idx >= len(index):
        return None  # horizon hasn't elapsed yet

    entry_price = float(closes.iloc[entry_idx])
    exit_price = float(closes.iloc[exit_idx])
    if entry_price <= 0:
        return None
    realized_return_pct = (exit_price / entry_price - 1.0) * 100

    outcome = None
    if signal == "Buy":
        outcome = "hit" if realized_return_pct > 0 else "miss"
    elif signal == "Trim":
        outcome = "hit" if realized_return_pct <= 0 else "miss"

    return {
        "entry_date": index[entry_idx].date().isoformat(),
        "exit_date": index[exit_idx].date().isoformat(),
        "realized_return_pct": round(realized_return_pct, 2),
        "outcome": outcome,
    }


def evaluate_signal_history(history: list[dict], closes: pd.Series) -> list[dict]:
    """DET-3: attaches short_outcome/long_outcome to each row of a
    ticker's own stock_scores history. The two scores have independent,
    non-overlapping horizons (SCR-1: 10-90 days vs 1-3 years), so each is
    evaluated separately against the same price series."""
    result = []
    for row in history:
        as_of = date.fromisoformat(row["as_of_date"])
        result.append(
            {
                **row,
                "short_outcome": evaluate_signal_outcome(as_of, row["short_signal"], closes, DET3_SHORT_HORIZON_DAYS),
                "long_outcome": evaluate_signal_outcome(as_of, row["long_signal"], closes, DET3_LONG_HORIZON_DAYS),
            }
        )
    return result
