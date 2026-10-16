"""COV-2: ETF scoring -- holdings-weighted scores, tracking error, and
real-holdings-based portfolio overlap, layered on top of the existing
Index Fund Screener (services/index_fund_service.py), which already covers
this app's curated ETF universe and its expense ratio / tracking
DIFFERENCE (CAGR gap) stats -- reused here, not rebuilt. "Tracking error"
(the volatility of the daily return gap, a different statistic from
"tracking difference") is new.

Two real data-source limits, disclosed rather than hidden:
  - yfinance's funds_data.top_holdings returns only each ETF's top ~10
    disclosed holdings, never its full holdings list (no free source for
    that) -- the holdings-weighted score and portfolio-overlap numbers are
    both scoped to those top holdings, with their own coverage % returned
    alongside so a thin-coverage number is never mistaken for a precise
    one.
  - Tracking error is only computed for the same subset of funds
    index_fund_service.py already trusts a clean Yahoo benchmark_index_
    ticker for (see IndexFundCandidate.benchmark_index_ticker's own
    comment) -- None ("N/A") for the rest, never guessed against an
    unrelated index.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import yfinance as yf

from services.cache_utils import ttl_cache
from services.index_fund_service import INDEX_FUND_UNIVERSE, IndexFundCandidate, get_index_fund_table
from services.yfinance_cache import get_cached_history

HOLDINGS_TTL_SECONDS = 60 * 60 * 24  # holdings composition changes slowly
# Below this, a "weighted score" would be computed from too thin a slice
# of the ETF's real holdings to mean anything -- None ("not enough
# coverage"), not a number quietly built from one or two covered names.
MIN_COVERAGE_PCT_FOR_SCORE = 50.0


def etf_universe() -> list[str]:
    return [f.ticker for f in INDEX_FUND_UNIVERSE]


def find_candidate(ticker: str) -> Optional[IndexFundCandidate]:
    ticker = ticker.upper()
    for f in INDEX_FUND_UNIVERSE:
        if f.ticker == ticker:
            return f
    return None


@ttl_cache(maxsize=64, ttl_seconds=HOLDINGS_TTL_SECONDS)
def fetch_etf_holdings(ticker: str) -> list[dict]:
    """Top ~10 disclosed holdings with weights, straight from Yahoo's own
    fund data -- there is no free source for an ETF's FULL holdings list,
    so every caller must treat the result as a sample, not the whole
    portfolio (see this module's docstring). [] on any failure or for a
    fund Yahoo doesn't expose holdings for (e.g. some bond funds)."""
    try:
        funds_data = yf.Ticker(ticker).funds_data
        top = funds_data.top_holdings if funds_data is not None else None
    except Exception:
        return []
    if top is None or top.empty:
        return []
    out = []
    for symbol, row in top.iterrows():
        weight = row.get("Holding Percent")
        if weight is None:
            continue
        out.append({"symbol": str(symbol).upper(), "name": row.get("Name"), "weight": float(weight)})
    return out


def compute_holdings_weighted_score(holdings: list[dict], scores_by_ticker: dict[str, float]) -> dict:
    """Pure: weights each top holding's own short_score (already fetched
    by the caller from stock_scores) by its disclosed holding weight,
    renormalized over the holdings this app actually scores -- a holding
    this app doesn't cover (most ETFs will have at least one, e.g. a
    foreign or small-cap name) is excluded, not guessed at."""
    if not holdings:
        return {"weighted_score": None, "coverage_pct": 0.0, "covered_holdings": 0, "total_holdings": 0}

    total_weight = sum(h["weight"] for h in holdings)
    covered = [h for h in holdings if h["symbol"] in scores_by_ticker]
    covered_weight = sum(h["weight"] for h in covered)
    coverage_pct = round((covered_weight / total_weight) * 100, 1) if total_weight else 0.0

    weighted_score = None
    if covered_weight > 0 and coverage_pct >= MIN_COVERAGE_PCT_FOR_SCORE:
        weighted_score = round(
            sum(scores_by_ticker[h["symbol"]] * h["weight"] for h in covered) / covered_weight, 1
        )

    return {
        "weighted_score": weighted_score,
        "coverage_pct": coverage_pct,
        "covered_holdings": len(covered),
        "total_holdings": len(holdings),
    }


def compute_tracking_error(ticker: str) -> Optional[float]:
    """Annualized std dev of (ETF daily return - benchmark daily return)
    over 1 year -- only for a fund with a confidently-mapped
    benchmark_index_ticker (same gate index_fund_service.py's own
    _tracking_difference uses); None ("N/A") otherwise, never guessed
    against an unrelated index."""
    candidate = find_candidate(ticker)
    if candidate is None or not candidate.benchmark_index_ticker:
        return None
    try:
        fund_hist = get_cached_history(ticker, "1y", auto_adjust=True)
        bench_hist = get_cached_history(candidate.benchmark_index_ticker, "1y", auto_adjust=True)
    except Exception:
        return None
    if fund_hist.empty or bench_hist.empty:
        return None
    fund_returns = fund_hist["Close"].pct_change().dropna()
    bench_returns = bench_hist["Close"].pct_change().dropna()
    common = fund_returns.index.intersection(bench_returns.index)
    if len(common) < 30:
        return None
    diff = fund_returns.loc[common] - bench_returns.loc[common]
    return round(float(diff.std() * np.sqrt(252) * 100), 2)


def compute_portfolio_overlap(holdings: list[dict], user_tickers: list[str]) -> dict:
    """Pure: % of this ETF's disclosed (top-N) holding weight that
    matches tickers the user already holds directly -- scoped to the same
    top-holdings sample fetch_etf_holdings returns, same disclosed limit."""
    if not holdings:
        return {"overlap_pct": 0.0, "overlapping_tickers": []}
    user_set = {t.upper() for t in user_tickers}
    overlapping = [h for h in holdings if h["symbol"] in user_set]
    overlap_pct = round(sum(h["weight"] for h in overlapping) * 100, 1)
    return {"overlap_pct": overlap_pct, "overlapping_tickers": [h["symbol"] for h in overlapping]}


def get_expense_ratios() -> dict[str, Optional[float]]:
    """Reuses index_fund_service.py's already-computed expense ratios
    (its own comment explains why netExpenseRatio is the only reliable
    field) for the whole universe in one call, rather than a second
    yfinance lookup per ticker."""
    df = get_index_fund_table()
    out: dict[str, Optional[float]] = {}
    for _, row in df.iterrows():
        value = row.get("Expense Ratio %")
        out[str(row["Ticker"]).upper()] = (
            float(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else None
        )
    return out
