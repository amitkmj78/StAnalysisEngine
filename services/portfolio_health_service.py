"""
Pure computation helpers for the Portfolio Health Check page
(docs/stock-analysis-requirements.html, HLT-1..4). No FastAPI/DB imports
here -- every function takes already-fetched data, matching services/
portfolio_compare_service.py's and services/stock_finder_service.py's own
convention, so this is fully testable with synthetic data.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from services.backtest_engine import max_drawdown_pct
from services.stock_finder_service import _gics_sector
from services.yfinance_cache import get_cached_history

BENCHMARK_TICKER = "SPY"

# Re-declared rather than imported from services/million_plan_service.py
# (which pulls in unrelated goal-plan-solver internals for one shared
# constant) -- same "small constant re-declared per domain" convention as
# portfolio_review_service.MAX_PARALLEL_REVIEW_FETCHES vs. index_fund_
# service's own constant. Reuses the exact three-value taxonomy that
# module already established for strategy_plans.account_type.
ACCOUNT_TYPES = ["Taxable", "Traditional", "Roth"]

# Each of these is a live network fetch per ticker/holding -- bounded the
# same way as every other multi-ticker fan-out in this codebase.
MAX_PARALLEL_HEALTH_FETCHES = 4


def _fetch_close_for_period(ticker: str, period: str) -> pd.Series:
    """period is a yfinance calendar-period string ("1y"/"3y"), distinct
    from portfolio_compare_service's trading-day window codes (that
    module tops out at "1Y"=252 trading days and its own _fetch_close
    caps history at 2y -- neither covers a real 3-year window), so this
    is new, deliberately parallel plumbing rather than a reuse."""
    hist = get_cached_history(ticker, period, auto_adjust=True)
    return hist["Close"] if not hist.empty else pd.Series(dtype=float)


def compute_portfolio_risk_metrics(positions: list[dict], period: str) -> dict:
    """positions: [{"ticker", "market_value"}]. Adapts stock_finder_
    service.compute_basket_risk_preview's covariance pattern (weighted-
    blend per-holding daily returns, np.cov for beta) -- adapted rather
    than called directly because that function expects a basket
    DataFrame shape (Ticker/Weight_pct/GICS Sector columns from a stock-
    finder scan) a real position list doesn't have, and it doesn't
    compute correlation at all (confirmed zero np.corrcoef usage
    anywhere in this repo).

    data_start/data_end are the REAL first/last dates in the aligned
    return series actually used, not just the requested period string --
    a holding with shorter history can shrink the effective window below
    the nominal "1y"/"3y" ask, and HLT-2 requires stating the period
    actually used, not the period requested."""
    if not positions:
        return {
            "period": period, "data_start": None, "data_end": None,
            "volatility_pct": None, "beta_to_spy": None, "correlation_to_spy": None,
            "max_drawdown_pct": None, "excluded_from_risk": [],
        }

    tickers = [p["ticker"] for p in positions]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HEALTH_FETCHES) as executor:
        closes_list = list(executor.map(lambda t: _fetch_close_for_period(t, period), tickers))
    closes = {t: s for t, s in zip(tickers, closes_list) if not s.empty}
    spy_close = _fetch_close_for_period(BENCHMARK_TICKER, period)

    excluded_from_risk = [t for t in tickers if t not in closes]
    if not closes or spy_close.empty:
        return {
            "period": period, "data_start": None, "data_end": None,
            "volatility_pct": None, "beta_to_spy": None, "correlation_to_spy": None,
            "max_drawdown_pct": None, "excluded_from_risk": excluded_from_risk,
        }

    price_df = pd.DataFrame({**closes, "__SPY__": spy_close}).dropna(how="any")
    kept_tickers = [t for t in closes if t in price_df.columns]
    excluded_from_risk += [t for t in tickers if t not in kept_tickers and t not in excluded_from_risk]

    if price_df.empty or len(kept_tickers) == 0:
        return {
            "period": period, "data_start": None, "data_end": None,
            "volatility_pct": None, "beta_to_spy": None, "correlation_to_spy": None,
            "max_drawdown_pct": None, "excluded_from_risk": excluded_from_risk,
        }

    returns_df = price_df[kept_tickers].pct_change().dropna()
    spy_returns = price_df["__SPY__"].pct_change().dropna()
    returns_df, spy_returns = returns_df.align(spy_returns, join="inner", axis=0)

    value_by_ticker = {p["ticker"]: p["market_value"] for p in positions}
    weights = pd.Series({t: value_by_ticker.get(t, 0.0) for t in kept_tickers})
    weights = weights / weights.sum() if weights.sum() else weights

    blended = (returns_df[kept_tickers] * weights).sum(axis=1)

    if blended.empty:
        return {
            "period": period, "data_start": None, "data_end": None,
            "volatility_pct": None, "beta_to_spy": None, "correlation_to_spy": None,
            "max_drawdown_pct": None, "excluded_from_risk": excluded_from_risk,
        }

    volatility_pct = float(blended.std() * np.sqrt(252) * 100)
    spy_aligned = spy_returns.loc[blended.index]
    if blended.var() and spy_aligned.var():
        cov = np.cov(blended, spy_aligned, ddof=1)
        beta_to_spy = float(cov[0, 1] / cov[1, 1])
        correlation_to_spy = float(np.corrcoef(blended, spy_aligned)[0, 1])
    else:
        beta_to_spy = None
        correlation_to_spy = None
    drawdown = max_drawdown_pct((blended * 100).tolist())

    return {
        "period": period,
        "data_start": blended.index[0].date().isoformat(),
        "data_end": blended.index[-1].date().isoformat(),
        "volatility_pct": round(volatility_pct, 2) if volatility_pct is not None else None,
        "beta_to_spy": round(beta_to_spy, 3) if beta_to_spy is not None else None,
        "correlation_to_spy": round(correlation_to_spy, 3) if correlation_to_spy is not None else None,
        "max_drawdown_pct": drawdown,
        "excluded_from_risk": excluded_from_risk,
    }


def compute_risk_over_windows(positions: list[dict]) -> dict:
    """{"1Y": compute_portfolio_risk_metrics(..., "1y"), "3Y": (..., "3y")}."""
    return {
        "1Y": compute_portfolio_risk_metrics(positions, "1y"),
        "3Y": compute_portfolio_risk_metrics(positions, "3y"),
    }


def compute_portfolio_sector_weights(positions: list[dict]) -> dict[str, float]:
    """positions: [{ticker, sector, market_value}]. Same grouping as
    portfolio_review_service.compute_sector_concentration but returns
    EVERY sector's weight (no 40%-flag floor) -- that function is a
    concentration flag, this is the full breakdown HLT-1's S&P-500
    comparison needs.

    Each position's sector is run through stock_finder_service._gics_
    sector() before grouping -- compute_sectors (the usual sector
    source) returns RAW Yahoo sector names (e.g. "Technology"), while
    stock_finder_service.compute_sp500_sector_mix() groups by GICS-
    renamed names (e.g. "Information Technology"). Comparing raw names
    against GICS names directly would silently produce two separate,
    non-overlapping rows for what's really the same sector -- this
    mapping is what keeps the two sides comparable.

    Positions with no sector (funds/ETFs, mostly) are excluded from the
    numerator but NOT the denominator, so a portfolio heavy in
    unclassified ETFs correctly shows sector weights that don't sum to
    100%, rather than a false 100%-of-classified-only picture."""
    total = sum(p.get("market_value") or 0.0 for p in positions)
    if total <= 0:
        return {}

    by_sector: dict[str, float] = {}
    for p in positions:
        raw_sector = p.get("sector")
        market_value = p.get("market_value")
        if not raw_sector or market_value is None:
            continue
        sector = _gics_sector(raw_sector)
        by_sector[sector] = by_sector.get(sector, 0.0) + market_value

    return {sector: round(value / total * 100.0, 2) for sector, value in by_sector.items()}


def build_sector_comparison(portfolio_weights: dict[str, float], sp500_weights: dict[str, float]) -> list[dict]:
    """[{sector, portfolio_weight_pct, sp500_weight_pct, gap_pct}], union
    of both sides' sectors -- a sector present on only one side shows 0.0
    on the other, rather than being dropped. gap_pct = portfolio - S&P
    500 (positive = overweight vs. the index)."""
    sectors = sorted(set(portfolio_weights) | set(sp500_weights))
    result = []
    for sector in sectors:
        p_weight = portfolio_weights.get(sector, 0.0)
        sp_weight = sp500_weights.get(sector, 0.0)
        result.append(
            {
                "sector": sector,
                "portfolio_weight_pct": round(p_weight, 2),
                "sp500_weight_pct": round(sp_weight, 2),
                "gap_pct": round(p_weight - sp_weight, 2),
            }
        )
    return result
