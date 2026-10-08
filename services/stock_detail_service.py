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


def past_earnings_dates(earnings_dates: pd.DataFrame, as_of: Optional[date] = None, limit: int = 8) -> list[dict]:
    """DET-4/ERN-2: already-reported earnings dates, most recent first.
    reported_eps/eps_estimate/surprise_pct all come straight off the same
    get_earnings_dates() frame next_earnings_date reads -- Surprise(%) is
    confirmed (via SCR-1's live yfinance verification) to already be a
    plain percent, so it's safe to surface now (earlier DET-4 work left
    it out pending that confirmation). eps_beat is None, not guessed,
    when either value is missing. revenue_beat is always None: yfinance
    has no historical revenue-estimate-vs-actual for past quarters
    anywhere (only forward-looking current-consensus snapshots exist),
    so this is a deliberate, disclosed gap rather than a fabricated or
    silently-omitted figure -- ERN-2's revenue half stays unavailable
    until a future PIT capture job exists to build that history going
    forward."""
    if earnings_dates.empty:
        return []
    index = earnings_dates.index
    if getattr(index, "tz", None) is not None:
        earnings_dates = earnings_dates.copy()
        earnings_dates.index = index.tz_localize(None)

    as_of_ts = pd.Timestamp(as_of or date.today())
    past = earnings_dates[earnings_dates.index.normalize() < as_of_ts].sort_index(ascending=False).head(limit)

    results = []
    for ts, row in past.iterrows():
        reported_eps = row.get("Reported EPS")
        eps_estimate = row.get("EPS Estimate")
        surprise_pct = row.get("Surprise(%)")
        reported_eps = None if reported_eps is None or pd.isna(reported_eps) else round(float(reported_eps), 2)
        eps_estimate = None if eps_estimate is None or pd.isna(eps_estimate) else round(float(eps_estimate), 2)
        eps_beat = None if reported_eps is None or eps_estimate is None else reported_eps > eps_estimate
        results.append(
            {
                "date": ts.date().isoformat(),
                "reported_eps": reported_eps,
                "eps_estimate": eps_estimate,
                "eps_beat": eps_beat,
                "surprise_pct": None if surprise_pct is None or pd.isna(surprise_pct) else round(float(surprise_pct), 2),
                "revenue_beat": None,
            }
        )
    return results


def _infer_market_timing(ts: pd.Timestamp) -> str:
    """ERN-1/ERN-2: yfinance has no explicit before/after-market flag
    anywhere (not in get_earnings_dates(), not in .calendar) -- inferred
    from the earnings timestamp's own time-of-day, confirmed live against
    real tickers (AAPL/MSFT always report 16:00 = after close; JPM/WMT
    show 06:00-08:00 = before open). This is a reasoned proxy, not a
    confirmed BMO/AMC flag from the data provider -- callers should
    disclose it as inferred."""
    return "before_market" if ts.hour < 12 else "after_market"


def next_day_move_pct(earnings_dates: pd.DataFrame, closes: pd.Series, as_of: Optional[date] = None, limit: int = 8) -> list[dict]:
    """ERN-2: the stock's price move around each of the last `limit`
    reported earnings dates. Market timing changes which two closes are
    compared: an after-market (AMC) report's reaction shows up in the
    *next* trading day's close, so entry=earnings-day close, exit=next
    day's close (mirrors evaluate_signal_outcome(horizon_days=1)). A
    before-market (BMO) report's reaction is already baked into that
    same day's close, so entry=prior trading day's close, exit=earnings-
    day close instead. move_pct is None per-row (not raised, not
    guessed) when the close series doesn't cover the needed index --
    e.g. a very recent earnings date with no next-day bar yet.

    tz_localize(None) only strips the tz label (same wall-clock time,
    same hour) -- same convention as next_earnings_date/past_earnings_
    dates above -- so market timing can be read off the hour after
    normalizing, no need to infer it from the pre-normalized index."""
    if earnings_dates.empty or closes.empty:
        return []

    dates = earnings_dates.copy()
    if getattr(dates.index, "tz", None) is not None:
        dates.index = dates.index.tz_localize(None)

    price_index = closes.index
    if getattr(price_index, "tz", None) is not None:
        closes = closes.copy()
        closes.index = price_index.tz_localize(None)
        price_index = closes.index

    as_of_ts = pd.Timestamp(as_of or date.today())
    past = dates[dates.index.normalize() < as_of_ts].sort_index(ascending=False).head(limit)

    results = []
    for ts, _row in past.iterrows():
        market_timing = _infer_market_timing(ts)
        # Price bars are date-indexed at midnight; ts carries a real
        # time-of-day (e.g. 16:00) -- compare by calendar date (normalize)
        # so the earnings day's own bar isn't skipped past.
        on_or_after = price_index[price_index >= ts.normalize()]
        move_pct = None
        if len(on_or_after) > 0:
            earnings_idx = price_index.get_loc(on_or_after[0])
            if market_timing == "after_market":
                entry_idx, exit_idx = earnings_idx, earnings_idx + 1
            else:
                entry_idx, exit_idx = earnings_idx - 1, earnings_idx
            if 0 <= entry_idx < len(price_index) and 0 <= exit_idx < len(price_index):
                entry_price = float(closes.iloc[entry_idx])
                exit_price = float(closes.iloc[exit_idx])
                if entry_price > 0:
                    move_pct = round((exit_price / entry_price - 1.0) * 100, 2)
        results.append({"date": ts.date().isoformat(), "market_timing": market_timing, "move_pct": move_pct})
    return results


def typical_earnings_move(moves: list[dict]) -> Optional[dict]:
    """ERN-3: the average *absolute* move size across whatever quarters
    next_day_move_pct could actually compute -- a +5% quarter and a -5%
    quarter both mean "usually moves about 5%", so this deliberately
    doesn't let direction cancel out. quarters_counted lets the caller
    caveat a small sample (e.g. a recent IPO with only 2-3 reported
    quarters) rather than implying a full 8-quarter average always
    exists."""
    computable = [m["move_pct"] for m in moves if m.get("move_pct") is not None]
    if not computable:
        return None
    avg_abs = sum(abs(m) for m in computable) / len(computable)
    return {"avg_abs_move_pct": round(avg_abs, 1), "quarters_counted": len(computable)}


def upcoming_earnings_in_window(earnings_dates: pd.DataFrame, as_of: Optional[date] = None, window_days: int = 30) -> Optional[dict]:
    """ERN-1: the next earnings date for this ticker if (and only if) it
    falls within the next `window_days` days -- None otherwise, an
    honest "nothing in this window" rather than reaching further out.
    Earnings occur roughly every ~13 weeks, so at most one row can ever
    match a 30-day window; no dedup logic needed."""
    if earnings_dates.empty:
        return None
    dates = earnings_dates.copy()
    if getattr(dates.index, "tz", None) is not None:
        dates.index = dates.index.tz_localize(None)

    as_of_ts = pd.Timestamp(as_of or date.today())
    window_end = as_of_ts + pd.Timedelta(days=window_days)
    upcoming = dates[(dates.index.normalize() >= as_of_ts) & (dates.index.normalize() <= window_end)].sort_index()
    if upcoming.empty:
        return None

    ts = upcoming.index[0]
    eps_estimate = upcoming.iloc[0].get("EPS Estimate")
    return {
        "date": ts.date().isoformat(),
        "market_timing": _infer_market_timing(ts),
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
        # FND-3: the public (all-tickers) track record persists these
        # alongside realized_return_pct, same display parity the old
        # rank-based signal_outcomes schema had -- additive fields, no
        # existing caller inspects dict equality (checked: only key lookups).
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
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
