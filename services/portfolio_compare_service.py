"""
Pure-function backbone for the rebuilt "Portfolio vs. Top Picks" compare
page: one call assembles the portfolio's window-relative return/
volatility/drawdown/contribution figures, a benchmark (SPY) comparison,
top-ranked funds for a goal, and a synthesized headline. No FastAPI/DB
imports here (matches services/benchmark_comparison_service.py and
services/backtest_engine.py's convention) so every function is testable
with synthetic pd.Series data, no live network or database required.

Deliberately NOT included (per explicit product decision, not an
oversight): a per-fund "Overlap %" against the user's holdings -- that
needs real per-stock fund constituent weights, which do not exist
anywhere in this app or its data sources (confirmed by a dedicated
research pass). Every response this module builds omits an overlap
field entirely rather than faking one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from services.backtest_engine import max_drawdown_pct
from services.index_fund_service import rank_funds_overall, rank_index_funds
from services.yfinance_cache import get_cached_history

BENCHMARK_TICKER = "SPY"
# Trading-day windows (NOT calendar days) -- matches web/backend/routers/
# momentum.py's existing WINDOWS={10,30,60,90} semantics, deliberately
# distinct from benchmark_comparison_service.py's calendar-day "since
# portfolio creation" convention, which this module doesn't touch.
WINDOW_TRADING_DAYS = {"10D": 10, "30D": 30, "60D": 60, "90D": 90, "1Y": 252}
REBASE_TO = 10_000.0
SPARK_POINTS = 14


@dataclass(frozen=True)
class WindowBounds:
    code: str
    start: pd.Timestamp
    end: pd.Timestamp
    trading_days: int


def resolve_window(window_code: str) -> WindowBounds:
    """
    Anchored to SPY's own cached price history as the reference clock
    (always available, long history, and every series in the response is
    aligned to it) -- same "identical trading days for every series"
    philosophy as index_fund_service._window_bounds, just anchored to one
    fixed reference series since here we control exactly what gets
    fetched, rather than intersecting several series' own last dates.
    """
    if window_code not in WINDOW_TRADING_DAYS:
        raise ValueError(f"window must be one of {sorted(WINDOW_TRADING_DAYS)}")
    n = WINDOW_TRADING_DAYS[window_code]
    ref = get_cached_history(BENCHMARK_TICKER, "2y", auto_adjust=True)["Close"].dropna()
    if len(ref) <= n:
        raise ValueError("Not enough SPY history to resolve this window.")
    start, end = ref.index[-(n + 1)], ref.index[-1]
    return WindowBounds(window_code, start, end, n)


def _slice(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return prices[(prices.index >= start) & (prices.index <= end)]


def _rebased_series(prices: pd.Series) -> list:
    """[[iso_date, value], ...], value = (price / price.iloc[0]) * 10000 --
    a "growth of $10,000" series. Nothing in the repo builds a
    date-indexed rebased series today; this is the one place that does."""
    if prices.empty:
        return []
    rebased = (prices / float(prices.iloc[0])) * REBASE_TO
    return [[d.date().isoformat(), round(float(v), 2)] for d, v in rebased.items()]


def _series_stats(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """return_pct, volatility_pct (annualized, same std*sqrt(252)*100
    formula as stock_finder_service.compute_basket_risk_preview), and the
    rebased series -- for one ticker's own price history sliced to the
    window (used for the benchmark and the #1 top fund)."""
    window_prices = _slice(prices, start, end)
    if len(window_prices) < 2:
        return {"return_pct": None, "volatility_pct": None, "series": []}
    return_pct = (float(window_prices.iloc[-1]) / float(window_prices.iloc[0]) - 1.0) * 100.0
    daily = window_prices.pct_change().dropna()
    volatility_pct = float(daily.std() * np.sqrt(252) * 100) if not daily.empty else None
    return {"return_pct": return_pct, "volatility_pct": volatility_pct, "series": _rebased_series(window_prices)}


@dataclass(frozen=True)
class HoldingInput:
    ticker: str
    shares: float
    current_price: Optional[float]
    acquired_at: Optional[date]


def _holding_window_return(h: HoldingInput, prices: Optional[pd.Series], bounds: WindowBounds) -> dict:
    """
    Window-relative return for one holding -- NOT a reuse of
    Unrealized_PnL_% (that's since-original-cost-basis, a different
    question). effective_start = max(window.start, acquired_at): a
    holding bought inside the window returns its real since-purchase
    return and carries `since` (an ISO date) so the UI can show "bought
    Aug 12" instead of silently mislabeling a partial-window return as a
    full-window one.
    """
    result: dict = {"ticker": h.ticker, "return_pct": None, "spark": [], "since": None}
    if prices is None or prices.empty:
        return result

    effective_start = bounds.start
    since: Optional[str] = None
    if h.acquired_at is not None:
        acquired_ts = pd.Timestamp(h.acquired_at)
        if acquired_ts > bounds.start:
            effective_start = acquired_ts
            since = h.acquired_at.isoformat()

    window_prices = _slice(prices, effective_start, bounds.end)
    if len(window_prices) >= 2:
        result["return_pct"] = (float(window_prices.iloc[-1]) / float(window_prices.iloc[0]) - 1.0) * 100.0
    result["spark"] = [round(float(v), 2) for v in window_prices.tail(SPARK_POINTS).tolist()]
    result["since"] = since
    return result


def build_portfolio_window_view(
    holdings: list[HoldingInput], bounds: WindowBounds, cash_balance: float = 0.0
) -> dict:
    """
    Retroactive-today's-weights approximation -- the same documented
    caveat as stock_finder_service.compute_basket_risk_preview ("as if
    today's picks/weights had been held, unchanged, for the whole
    window"), applied here to a user's real holdings. weight = market
    value / total market value using TODAY's prices (per spec). Cash
    counts as a 0%-return holding (per spec) when cash_balance > 0.

    portfolio_return_pct = sum(contribution_pts) BY CONSTRUCTION -- it's
    literally the sum, so the identity holds exactly, not approximately;
    any tolerance needed in a test/acceptance-check is just float-
    rounding slack on the persisted round(...,4) value, not a deeper
    property to verify.

    Returns {return_pct, volatility_pct, max_drawdown_pct, series,
    holdings, excluded_from_risk}.
    """
    market_values = {h.ticker: (h.shares or 0) * (h.current_price or 0) for h in holdings}
    total_value = sum(market_values.values()) + max(cash_balance, 0.0)

    closes: dict[str, pd.Series] = {}
    excluded: list[str] = []
    for h in holdings:
        hist = get_cached_history(h.ticker, "2y", auto_adjust=True)
        s = hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float)
        if s.empty:
            excluded.append(h.ticker)
        else:
            closes[h.ticker] = s

    per_holding = []
    for h in holdings:
        weight_pct = (market_values[h.ticker] / total_value * 100.0) if total_value else 0.0
        row = _holding_window_return(h, closes.get(h.ticker), bounds)
        row["weight_pct"] = round(weight_pct, 2)
        row["contribution_pts"] = round(weight_pct / 100.0 * (row["return_pct"] or 0.0), 4)
        per_holding.append(row)

    if cash_balance > 0:
        cash_weight_pct = cash_balance / total_value * 100.0 if total_value else 0.0
        per_holding.append({
            "ticker": "CASH", "return_pct": 0.0, "spark": [], "since": None,
            "weight_pct": round(cash_weight_pct, 2), "contribution_pts": 0.0,
        })

    portfolio_return_pct = sum(r["contribution_pts"] for r in per_holding) if per_holding else None

    # Blended daily-return series for volatility/drawdown/rebased chart --
    # weight-blended the same way compute_basket_risk_preview does, but
    # from WINDOW-sliced prices, not a fixed lookback. Weights renormalize
    # over just the tickers with real price history (excluded_from_risk
    # dropped), same "don't let a data gap silently zero out the whole
    # blend" approach used there.
    if closes:
        price_df = pd.DataFrame({t: _slice(s, bounds.start, bounds.end) for t, s in closes.items()}).sort_index()
        returns_df = price_df.pct_change().fillna(0.0)
        weights = pd.Series({t: market_values[t] for t in closes})
        weights = weights / weights.sum() if weights.sum() else weights
        blended_returns = (returns_df[weights.index] * weights).sum(axis=1)
    else:
        blended_returns = pd.Series(dtype=float)

    if not blended_returns.empty:
        volatility_pct = float(blended_returns.std() * np.sqrt(252) * 100)
        drawdown_pct = max_drawdown_pct((blended_returns * 100).tolist())
        rebased_prices = (1 + blended_returns).cumprod() * REBASE_TO
        rebased_prices.iloc[0] = REBASE_TO  # anchor exactly at 10000 on day 1, not 1+first-day drift
        series = [[d.date().isoformat(), round(float(v), 2)] for d, v in rebased_prices.items()]
    else:
        volatility_pct = None
        drawdown_pct = None
        series = []

    return {
        "return_pct": portfolio_return_pct,
        "volatility_pct": volatility_pct,
        "max_drawdown_pct": drawdown_pct,
        "series": series,
        "holdings": per_holding,
        "excluded_from_risk": excluded,
    }


def select_gap_drivers(holdings: list[dict]) -> list[dict]:
    """Largest positive contribution ("lead") + two most negative
    ("drag"). Ties broken by ticker alphabetically for determinism."""
    positive = sorted(
        (h for h in holdings if (h["contribution_pts"] or 0) > 0),
        key=lambda h: (-h["contribution_pts"], h["ticker"]),
    )
    negative = sorted(
        (h for h in holdings if (h["contribution_pts"] or 0) < 0),
        key=lambda h: (h["contribution_pts"], h["ticker"]),
    )
    drivers = []
    if positive:
        drivers.append({"ticker": positive[0]["ticker"], "kind": "lead", "contribution_pts": positive[0]["contribution_pts"]})
    for h in negative[:2]:
        drivers.append({"ticker": h["ticker"], "kind": "drag", "contribution_pts": h["contribution_pts"]})
    return drivers


def build_headline(
    window_label: str,
    portfolio_return_pct: Optional[float],
    benchmark_return_pct: Optional[float],
    top_fund_ticker: Optional[str],
    top_fund_return_pct: Optional[float],
) -> str:
    """Pure string formatting, no LLM -- same "synthesize what's already
    shown" boundary as benchmark_comparison_service's `suggestion` field.
    Degrades gracefully when a piece is missing (still-loading data, an
    empty portfolio, or no top fund for the goal)."""
    if portfolio_return_pct is None:
        return "Not enough price history yet to summarize this window."
    verb = "up" if portfolio_return_pct >= 0 else "down"
    parts = [f"Your portfolio is {verb} {abs(portfolio_return_pct):.1f}% over {window_label}"]
    clauses = []
    if benchmark_return_pct is not None:
        gap = portfolio_return_pct - benchmark_return_pct
        clauses.append(f"{abs(gap):.1f} pts {'ahead of' if gap >= 0 else 'behind'} the S&P 500")
    if top_fund_ticker and top_fund_return_pct is not None:
        gap = portfolio_return_pct - top_fund_return_pct
        clauses.append(f"{abs(gap):.1f} pts {'ahead of' if gap >= 0 else 'behind'} {top_fund_ticker}")
    if clauses:
        parts.append(" — " + " and ".join(clauses))
    return "".join(parts) + "."


def derive_confidence(stability: Optional[dict]) -> dict:
    """
    Maps web/backend/pit_prices.py's get_signal_stability_for_ticker
    result ({"flip_count", "days_captured", "current_streak_days",
    "unstable"}, or None when there isn't enough capture history yet) to
    a coarse confidence label/score -- an explicit product decision to
    derive confidence from existing signal-stability data rather than
    add a new model output. score = 100 - flip_count*25 (clamped >=0,
    and to <=25 when already flagged unstable); label thresholds >=75
    high, 40-74 medium, <40 low, so score and label never disagree.
    """
    if stability is None:
        return {"label": "unknown", "score": None}
    score = max(0, 100 - stability["flip_count"] * 25)
    if stability["unstable"]:
        score = min(score, 25)
    label = "high" if score >= 75 else "medium" if score >= 40 else "low"
    return {"label": label, "score": score}


def _fund_reason(breakdown: Optional[dict]) -> str:
    """One-line reason a fund ranked where it did, from the same
    per-bucket _breakdown already computed by index_fund_service's
    _score_group -- reusing data already computed, not inventing new
    scoring just to explain the existing one."""
    if not breakdown:
        return "Ranks well for this goal."
    top_bucket = max(breakdown.items(), key=lambda kv: kv[1]["sub_score"])[0]
    return f"Leads on {top_bucket} for this goal."


def select_top_funds(goal: str, bounds: WindowBounds, top_n: int = 5) -> list[dict]:
    """
    WHICH funds rank #1-N depends only on `goal` (rank_index_funds/
    rank_funds_overall, both already used elsewhere in this app) -- the
    goal's own scoring windows (return_1y/3y/30d/60d/90d, etc.) are fixed
    internally by GOAL_WEIGHTS and independent of this page's window
    selector. The page's window only changes the window-relative
    return_pct/volatility_pct/series shown for those SAME funds (via
    _series_stats), so they're directly comparable to the portfolio's
    own window-relative figures. `series` is populated for rank 1 only
    (per spec -- the chart only plots portfolio/benchmark/top fund).
    """
    df, _ = rank_index_funds(goal, "All", "5y")
    if df.empty:
        return []
    ranked = rank_funds_overall(df).head(top_n).reset_index(drop=True)

    out = []
    for i, row in ranked.iterrows():
        ticker = row["Ticker"]
        hist = get_cached_history(ticker, "2y", auto_adjust=True)
        stats = _series_stats(hist["Close"].dropna() if not hist.empty else pd.Series(dtype=float), bounds.start, bounds.end)
        expense_ratio = row.get("Expense Ratio %")
        out.append({
            "rank": i + 1,
            "ticker": ticker,
            "name": row["Fund"],
            "score": float(row["Score"]),
            "reason": _fund_reason(row.get("_breakdown")),
            "return_pct": stats["return_pct"],
            "expense_ratio_pct": float(expense_ratio) if expense_ratio is not None and pd.notna(expense_ratio) else None,
            "volatility_pct": stats["volatility_pct"],
            "series": stats["series"] if i == 0 else None,
        })
    return out


def mark_owned(stock_rows: list[dict], portfolio_tickers: set[str]) -> list[dict]:
    for row in stock_rows:
        row["owned"] = row["ticker"] in portfolio_tickers
    return stock_rows
