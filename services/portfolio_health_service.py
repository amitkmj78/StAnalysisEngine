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
from services.yfinance_cache import get_cached_fund_top_holdings, get_cached_history

BENCHMARK_TICKER = "SPY"

# HLT-1: yfinance only discloses a fund's top 10 holdings (confirmed live:
# SPY's top 10 sum to ~37.8% of the fund, QQQ's to ~46.3% -- the rest is
# unavailable from any data source). Reused verbatim in both the API
# response and the frontend copy so the two never drift apart.
TOP10_DISCLOSURE = (
    "Overlap computed against each fund's top 10 holdings only — smaller "
    "positions inside the fund aren't visible."
)

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
    is new, deliberately parallel plumbing rather than a reuse.

    Unlike every other yfinance_cache function, get_cached_history does
    NOT fail open -- it can raise straight through (confirmed live: a
    rate-limited ticker raised yfinance.exceptions.YFRateLimitError
    here, 500-ing the whole risk endpoint over one bad ticker in a
    bounded ThreadPoolExecutor fan-out). Caught here instead so one
    ticker's fetch failure degrades to the same honest
    excluded_from_risk path as "no history for this ticker", not a
    crash."""
    try:
        hist = get_cached_history(ticker, period, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)
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


def fetch_fund_holdings_map(tickers: list[str]) -> dict[str, dict[str, float]]:
    """{fund_ticker: {underlying_symbol: weight_fraction}} -- ONLY for
    tickers where get_cached_fund_top_holdings returned non-empty. A
    ticker absent from this dict's keys is being treated as a plain
    stock (or a fund with nothing disclosed, e.g. GLD/BITO --
    functionally the same for look-through purposes: nothing to
    decompose either way). Bounded fan-out, same rationale as every
    other per-ticker yfinance call in this codebase."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HEALTH_FETCHES) as executor:
        holdings_list = list(executor.map(get_cached_fund_top_holdings, tickers))
    result = {}
    for ticker, holdings_df in zip(tickers, holdings_list):
        if holdings_df is None or holdings_df.empty or "Holding Percent" not in holdings_df.columns:
            continue
        result[ticker] = {str(symbol): float(pct) for symbol, pct in holdings_df["Holding Percent"].items()}
    return result


def compute_fund_coverage_pct(fund_holdings: dict[str, dict[str, float]]) -> dict[str, float]:
    """{fund_ticker: sum(weight_fraction) * 100} -- "top 10 holdings = X%
    of the fund," drives the per-fund disclosure copy (confirmed live:
    SPY ~=37.8%, QQQ ~=46.3%)."""
    return {ticker: round(sum(weights.values()) * 100, 1) for ticker, weights in fund_holdings.items()}


def compute_look_through_exposure(positions: list[dict], fund_holdings: dict[str, dict[str, float]]) -> list[dict]:
    """positions: [{ticker, market_value}]. For a fund position (a key in
    fund_holdings), its market_value is decomposed across the fund's
    disclosed holdings into look-through dollars per underlying symbol;
    the UNDISCLOSED remainder (1 - sum of disclosed fractions) stays
    attributed to the fund ticker itself -- dollars are always
    conserved, nothing silently dropped. A non-fund position's full
    market_value counts as direct exposure to itself. A symbol held
    both directly AND via one or more funds gets exactly ONE combined
    row (direct_value + look_through_value = combined_value) -- this is
    HLT-1's literal acceptance criterion ("a stock held directly and
    through ETFs is combined").

    Returns, sorted by combined_value descending:
    [{"ticker", "direct_value", "look_through_value", "combined_value",
      "combined_weight_pct", "via_funds": [{"fund_ticker", "dollars"}]}]
    """
    total_value = sum(p.get("market_value") or 0.0 for p in positions)
    direct_value: dict[str, float] = {}
    look_through_value: dict[str, float] = {}
    via_funds: dict[str, list[dict]] = {}

    for p in positions:
        ticker = p["ticker"]
        market_value = p.get("market_value") or 0.0
        holdings = fund_holdings.get(ticker)
        if holdings:
            disclosed_fraction = sum(holdings.values())
            for symbol, weight_fraction in holdings.items():
                dollars = market_value * weight_fraction
                look_through_value[symbol] = look_through_value.get(symbol, 0.0) + dollars
                via_funds.setdefault(symbol, []).append({"fund_ticker": ticker, "dollars": round(dollars, 2)})
            undisclosed_dollars = market_value * (1.0 - disclosed_fraction)
            direct_value[ticker] = direct_value.get(ticker, 0.0) + undisclosed_dollars
        else:
            direct_value[ticker] = direct_value.get(ticker, 0.0) + market_value

    all_tickers = set(direct_value) | set(look_through_value)
    rows = []
    for ticker in all_tickers:
        d = round(direct_value.get(ticker, 0.0), 2)
        lt = round(look_through_value.get(ticker, 0.0), 2)
        combined = round(d + lt, 2)
        rows.append(
            {
                "ticker": ticker,
                "direct_value": d,
                "look_through_value": lt,
                "combined_value": combined,
                "combined_weight_pct": round(combined / total_value * 100, 2) if total_value else 0.0,
                "via_funds": via_funds.get(ticker, []),
            }
        )
    rows.sort(key=lambda r: r["combined_value"], reverse=True)
    return rows


def _infer_annual_payment_count(dividends: pd.Series, lookback_days: int = 400) -> int:
    """Counts distinct payments in the trailing ~400 days (from the
    series' own most recent payment, not "today" -- this is historical
    data), and maps to an assumed payments-per-year frequency: >=10 ->
    12 (monthly), >=3 -> 4 (quarterly), ==2 -> 2 (semiannual), ==1 -> 1
    (annual), 0 -> 0. 400 rather than 365 gives quarterly/monthly payers
    a little slack so a payment landing just past the exact-365-day mark
    isn't missed and misread as a lower frequency."""
    if dividends.empty:
        return 0
    cutoff = dividends.index.max() - pd.Timedelta(days=lookback_days)
    count = int((dividends.index >= cutoff).sum())
    if count >= 10:
        return 12
    if count >= 3:
        return 4
    if count == 2:
        return 2
    if count == 1:
        return 1
    return 0


def compute_trailing_dividend_per_share(dividends: pd.Series) -> Optional[float]:
    """Sum of the last N payments, N = _infer_annual_payment_count --
    fixes a confirmed-live bug where a naive 365-day cutoff over-counts
    by one extra payment for issuers whose pay dates don't land exactly
    365 days apart (AAPL: naive sum 1.32 vs. Yahoo's own
    trailingAnnualDividendRate 1.05, ~26% inflation from one stray extra
    quarterly payment). None (not 0.0) when there's no payment history
    at all -- "never paid" and "we don't know" are different facts,
    same convention as services.stock_detail_service.recent_dividends."""
    n = _infer_annual_payment_count(dividends)
    if n == 0:
        return None
    return round(float(dividends.sort_index().tail(n).sum()), 4)


def compute_portfolio_dividend_income(
    positions: list[dict], dividends_by_ticker: dict[str, pd.Series], info_by_ticker: dict[str, dict]
) -> dict:
    """positions: [{ticker, shares}]. Pure function -- the caller (router)
    does the live fetches and passes the results in, same separation
    compute_look_through_exposure already uses for fund_holdings.
    Trailing income = compute_trailing_dividend_per_share(...) * shares.
    Projected income uses info.get("dividendRate") directly (Yahoo's own
    forward-looking per-share rate) -- not recomputed from the trailing
    series, since a forward rate and a trailing sum measure different
    things and shouldn't be conflated. A ticker with no trailing
    history or no dividendRate contributes None (not 0) to its own
    row, excluded from (not zeroing) the portfolio total."""
    by_ticker = []
    total_trailing = 0.0
    total_projected = 0.0
    any_trailing = False
    any_projected = False

    for p in positions:
        ticker = p["ticker"]
        shares = p.get("shares") or 0
        dividends = dividends_by_ticker.get(ticker, pd.Series(dtype=float))
        info = info_by_ticker.get(ticker) or {}

        trailing_per_share = compute_trailing_dividend_per_share(dividends)
        trailing_income = round(trailing_per_share * shares, 2) if trailing_per_share is not None else None

        projected_per_share = info.get("dividendRate")
        projected_income = round(projected_per_share * shares, 2) if projected_per_share is not None else None

        by_ticker.append(
            {
                "ticker": ticker,
                "trailing_per_share": trailing_per_share,
                "trailing_income": trailing_income,
                "projected_per_share": projected_per_share,
                "projected_income": projected_income,
            }
        )
        if trailing_income is not None:
            total_trailing += trailing_income
            any_trailing = True
        if projected_income is not None:
            total_projected += projected_income
            any_projected = True

    return {
        "by_ticker": by_ticker,
        "total_trailing_income": round(total_trailing, 2) if any_trailing else None,
        "total_projected_income": round(total_projected, 2) if any_projected else None,
    }


def compute_fee_drag(
    positions: list[dict], fund_holdings: dict[str, dict[str, float]], info_by_ticker: dict[str, dict]
) -> dict:
    """positions: [{ticker, market_value}]. A position counts as a fund
    for fee-drag purposes iff it's a key in fund_holdings -- the SAME
    classification compute_look_through_exposure already uses, not
    re-detected here. expense_ratio_pct = info["netExpenseRatio"] --
    confirmed live this is already a percentage-point value (0.0945
    means 0.0945%, not a fraction of 1), same convention already used in
    services/index_fund_service.py and GET /compare's benchmark section
    -- do NOT run it through a fraction-of-1 heuristic.
    annual_fee_drag_dollars = market_value * (expense_ratio_pct / 100).
    A fund with a missing netExpenseRatio gets null drag, excluded from
    the total, not treated as 0 (an unknown fee is not the same fact as
    a genuinely zero-cost fund)."""
    by_fund = []
    total = 0.0
    any_known = False

    for p in positions:
        ticker = p["ticker"]
        if ticker not in fund_holdings:
            continue
        market_value = p.get("market_value") or 0.0
        info = info_by_ticker.get(ticker) or {}
        expense_ratio_pct = info.get("netExpenseRatio")

        if expense_ratio_pct is None:
            annual_fee_drag_dollars = None
        else:
            annual_fee_drag_dollars = round(market_value * (expense_ratio_pct / 100.0), 2)
            total += annual_fee_drag_dollars
            any_known = True

        by_fund.append(
            {
                "ticker": ticker,
                "market_value": round(market_value, 2),
                "expense_ratio_pct": expense_ratio_pct,
                "annual_fee_drag_dollars": annual_fee_drag_dollars,
            }
        )

    return {
        "by_fund": by_fund,
        "total_annual_fee_drag_dollars": round(total, 2) if any_known else None,
    }


# HLT-4: this app has no transaction/lot history anywhere in its schema
# (portfolio_positions is a current-snapshot table only -- shares/
# avg_cost, never a buy/sell log), so whether the user has already
# repurchased a ticker within the wash-sale window is genuinely not
# computable. This generic, always-shown disclosure is the honest
# alternative to a fabricated or silently-omitted personalized check.
WASH_SALE_DISCLOSURE = (
    "This app has no transaction/lot history, so it cannot tell whether you've already "
    "repurchased a ticker recently. If you buy this ticker again within 30 days before or "
    "after selling it at a loss, the IRS wash-sale rule may disallow the loss for tax purposes."
)


def find_tax_loss_harvest_candidates(positions: list[dict], account_type: str) -> dict:
    """positions: [{ticker, shares, avg_cost, current_price,
    unrealized_pnl_pct}] -- straight from portfolio_positions, no new
    computation needed for the core filter (unrealized_pnl_pct already
    exists and already tells us which positions are below cost). Only
    returns candidates when account_type == "Taxable" (HLT-4's
    acceptance criterion) -- {"eligible": False, "candidates": []} with
    a stated reason otherwise, not silent emptiness. Candidates sorted
    by unrealized_loss_dollars ascending (biggest loss first)."""
    if account_type != "Taxable":
        return {
            "eligible": False,
            "reason": f"Tax-loss harvesting only applies to taxable accounts -- this portfolio is marked {account_type}.",
            "candidates": [],
        }

    candidates = []
    for p in positions:
        avg_cost = p.get("avg_cost")
        current_price = p.get("current_price")
        shares = p.get("shares")
        if avg_cost is None or current_price is None or shares is None:
            continue
        unrealized_pnl_pct = p.get("unrealized_pnl_pct")
        if unrealized_pnl_pct is None:
            unrealized_pnl_pct = (current_price / avg_cost - 1.0) * 100 if avg_cost else None
        if unrealized_pnl_pct is None or unrealized_pnl_pct >= 0:
            continue
        unrealized_loss_dollars = round((current_price - avg_cost) * shares, 2)
        candidates.append(
            {
                "ticker": p["ticker"],
                "shares": shares,
                "avg_cost": avg_cost,
                "current_price": current_price,
                "unrealized_loss_pct": round(unrealized_pnl_pct, 2),
                "unrealized_loss_dollars": unrealized_loss_dollars,
                "wash_sale_note": WASH_SALE_DISCLOSURE,
            }
        )
    candidates.sort(key=lambda c: c["unrealized_loss_dollars"])

    return {"eligible": True, "reason": None, "candidates": candidates}
