from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.cache_utils import ttl_cache
from services.yfinance_cache import get_cached_history, get_cached_info

# Was 10 — even with fetch_with_backoff's per-call pacing, 10 concurrent
# workers each making 3 calls was still a real, observed trigger for
# sustained Yahoo rate limiting.
MAX_PARALLEL_FETCHES = 4


@dataclass(frozen=True)
class IndexFundCandidate:
    ticker: str
    name: str
    benchmark: str
    category: str


INDEX_FUND_UNIVERSE: List[IndexFundCandidate] = [
    # US Large Blend
    IndexFundCandidate("VOO", "Vanguard S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("IVV", "iShares Core S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("SPLG", "SPDR Portfolio S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("SPY", "SPDR S&P 500 ETF Trust", "S&P 500", "US Large Blend"),
    # US Total Market
    IndexFundCandidate("VTI", "Vanguard Total Stock Market ETF", "CRSP US Total Market", "US Total Market"),
    IndexFundCandidate("ITOT", "iShares Core S&P Total US Stock Market ETF", "S&P Total US Stock Market", "US Total Market"),
    IndexFundCandidate("SCHB", "Schwab US Broad Market ETF", "Dow Jones US Broad Stock Market", "US Total Market"),
    # US Large Growth
    IndexFundCandidate("QQQ", "Invesco QQQ Trust", "Nasdaq-100", "US Large Growth"),
    IndexFundCandidate("VUG", "Vanguard Growth ETF", "CRSP US Large Cap Growth", "US Large Growth"),
    IndexFundCandidate("IWF", "iShares Russell 1000 Growth ETF", "Russell 1000 Growth", "US Large Growth"),
    IndexFundCandidate("SCHG", "Schwab US Large-Cap Growth ETF", "Dow Jones US Large-Cap Growth", "US Large Growth"),
    # US Large Value
    IndexFundCandidate("VTV", "Vanguard Value ETF", "CRSP US Large Cap Value", "US Large Value"),
    IndexFundCandidate("IWD", "iShares Russell 1000 Value ETF", "Russell 1000 Value", "US Large Value"),
    IndexFundCandidate("SCHV", "Schwab US Large-Cap Value ETF", "Dow Jones US Large-Cap Value", "US Large Value"),
    # US Mid Cap
    IndexFundCandidate("VO", "Vanguard Mid-Cap ETF", "CRSP US Mid Cap", "US Mid Cap"),
    IndexFundCandidate("IJH", "iShares Core S&P Mid-Cap ETF", "S&P MidCap 400", "US Mid Cap"),
    IndexFundCandidate("SCHM", "Schwab US Mid-Cap ETF", "Dow Jones US Mid-Cap", "US Mid Cap"),
    # US Small Cap
    IndexFundCandidate("IWM", "iShares Russell 2000 ETF", "Russell 2000", "US Small Cap"),
    IndexFundCandidate("VB", "Vanguard Small-Cap ETF", "CRSP US Small Cap", "US Small Cap"),
    IndexFundCandidate("IJR", "iShares Core S&P Small-Cap ETF", "S&P SmallCap 600", "US Small Cap"),
    IndexFundCandidate("SCHA", "Schwab US Small-Cap ETF", "Dow Jones US Small-Cap", "US Small Cap"),
    # International Developed
    IndexFundCandidate("VEA", "Vanguard FTSE Developed Markets ETF", "FTSE Developed All Cap ex US", "International Developed"),
    IndexFundCandidate("SCHF", "Schwab International Equity ETF", "FTSE Developed ex US", "International Developed"),
    IndexFundCandidate("IEFA", "iShares Core MSCI EAFE ETF", "MSCI EAFE IMI", "International Developed"),
    # International Total
    IndexFundCandidate("VXUS", "Vanguard Total International Stock ETF", "FTSE Global All Cap ex US", "International Total"),
    IndexFundCandidate("IXUS", "iShares Core MSCI Total International Stock ETF", "MSCI ACWI ex USA IMI", "International Total"),
    # Emerging Markets
    IndexFundCandidate("VWO", "Vanguard FTSE Emerging Markets ETF", "FTSE Emerging Markets All Cap China A Inclusion", "Emerging Markets"),
    IndexFundCandidate("IEMG", "iShares Core MSCI Emerging Markets ETF", "MSCI Emerging Markets IMI", "Emerging Markets"),
    IndexFundCandidate("SCHE", "Schwab Emerging Markets Equity ETF", "FTSE Emerging", "Emerging Markets"),
    # Bond — Total Market
    IndexFundCandidate("BND", "Vanguard Total Bond Market ETF", "Bloomberg US Aggregate Float Adjusted", "Bond — Total Market"),
    IndexFundCandidate("AGG", "iShares Core US Aggregate Bond ETF", "Bloomberg US Aggregate Bond", "Bond — Total Market"),
    # Bond — Short-Term
    IndexFundCandidate("BSV", "Vanguard Short-Term Bond ETF", "Bloomberg US 1-5yr Government/Credit Float Adjusted", "Bond — Short-Term"),
    IndexFundCandidate("SCHO", "Schwab Short-Term US Treasury ETF", "Bloomberg US Treasury 1-3 Year", "Bond — Short-Term"),
    IndexFundCandidate("VGSH", "Vanguard Short-Term Treasury ETF", "Bloomberg US Treasury 1-3 Year", "Bond — Short-Term"),
    # Bond — Long-Term/Treasury
    IndexFundCandidate("TLT", "iShares 20+ Year Treasury Bond ETF", "ICE US Treasury 20+ Year", "Bond — Long-Term/Treasury"),
    IndexFundCandidate("VGLT", "Vanguard Long-Term Treasury ETF", "Bloomberg US Long Treasury", "Bond — Long-Term/Treasury"),
    IndexFundCandidate("SPTL", "SPDR Portfolio Long Term Treasury ETF", "Bloomberg US Long Treasury", "Bond — Long-Term/Treasury"),
    # Bond — Corporate
    IndexFundCandidate("LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "Markit iBoxx USD Liquid Investment Grade", "Bond — Corporate"),
    IndexFundCandidate("VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "Bloomberg US 5-10yr Corporate", "Bond — Corporate"),
    # Bond — High Yield
    IndexFundCandidate("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "Markit iBoxx USD Liquid High Yield", "Bond — High Yield"),
    IndexFundCandidate("JNK", "SPDR Bloomberg High Yield Bond ETF", "Bloomberg Very Liquid High Yield", "Bond — High Yield"),
    # Bond — TIPS
    IndexFundCandidate("TIP", "iShares TIPS Bond ETF", "Bloomberg US TIPS", "Bond — TIPS"),
    IndexFundCandidate("SCHP", "Schwab US TIPS ETF", "Bloomberg US Treasury Inflation Protected Securities", "Bond — TIPS"),
    # Dividend/Income
    IndexFundCandidate("VYM", "Vanguard High Dividend Yield ETF", "FTSE High Dividend Yield", "Dividend/Income"),
    IndexFundCandidate("SCHD", "Schwab US Dividend Equity ETF", "Dow Jones US Dividend 100", "Dividend/Income"),
    IndexFundCandidate("DVY", "iShares Select Dividend ETF", "Dow Jones US Select Dividend", "Dividend/Income"),
    IndexFundCandidate("VIG", "Vanguard Dividend Appreciation ETF", "S&P US Dividend Growers", "Dividend/Income"),
    # Real Estate
    IndexFundCandidate("VNQ", "Vanguard Real Estate ETF", "MSCI US Investable Market Real Estate 25/50", "Real Estate"),
    IndexFundCandidate("SCHH", "Schwab US REIT ETF", "Dow Jones US Select REIT", "Real Estate"),
    # Sector
    IndexFundCandidate("XLK", "Technology Select Sector SPDR Fund", "Technology Select Sector", "Sector"),
    IndexFundCandidate("VGT", "Vanguard Information Technology ETF", "MSCI US Investable Market Information Technology 25/50", "Sector"),
    IndexFundCandidate("XLF", "Financial Select Sector SPDR Fund", "Financial Select Sector", "Sector"),
    IndexFundCandidate("VFH", "Vanguard Financials ETF", "MSCI US Investable Market Financials 25/50", "Sector"),
    IndexFundCandidate("XLV", "Health Care Select Sector SPDR Fund", "Health Care Select Sector", "Sector"),
    IndexFundCandidate("VHT", "Vanguard Health Care ETF", "MSCI US Investable Market Health Care 25/50", "Sector"),
    IndexFundCandidate("XLE", "Energy Select Sector SPDR Fund", "Energy Select Sector", "Sector"),
    IndexFundCandidate("VDE", "Vanguard Energy ETF", "MSCI US Investable Market Energy 25/50", "Sector"),
    IndexFundCandidate("XLY", "Consumer Discretionary Select Sector SPDR Fund", "Consumer Discretionary Select Sector", "Sector"),
    IndexFundCandidate("XLP", "Consumer Staples Select Sector SPDR Fund", "Consumer Staples Select Sector", "Sector"),
]


GOAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Balanced Core": {
        "return_1y": 0.35,
        "return_3y_annualized": 0.25,
        "expense_ratio": 0.20,
        "volatility_1y": 0.10,
        "max_drawdown_3y": 0.10,
    },
    "Lowest Cost": {
        "expense_ratio": 0.65,
        "return_3y_annualized": 0.20,
        "volatility_1y": 0.10,
        "assets_billions": 0.05,
    },
    # Deliberately NOT just "Balanced Core with return weighted higher" —
    # that made the two goals pick nearly identical top funds in practice
    # (both were return-dominated, and the same few funds led on every
    # return window). This instead measures recent momentum (30/60/90d)
    # rather than the 1Y/3Y windows Balanced Core already covers, so the
    # two goals actually answer different questions.
    "Best Growth": {
        "return_30d": 0.30,
        "return_60d": 0.25,
        "return_90d": 0.20,
        "return_1y": 0.15,
        "volatility_1y": 0.10,
    },
    "Most Stable": {
        "volatility_1y": 0.45,
        "max_drawdown_3y": 0.30,
        "expense_ratio": 0.15,
        "return_3y_annualized": 0.10,
    },
}


LOWER_IS_BETTER = {"expense_ratio", "volatility_1y", "max_drawdown_3y"}

METRIC_LABELS: Dict[str, str] = {
    "return_1y": "1-Year Return",
    "return_3y_annualized": "3-Year Annualized Return",
    "expense_ratio": "Expense Ratio",
    "volatility_1y": "1-Year Volatility",
    "max_drawdown_3y": "3-Year Max Drawdown",
    "assets_billions": "Fund Assets",
    "return_30d": "30-Day Return",
    "return_60d": "60-Day Return",
    "return_90d": "90-Day Return",
}

METRIC_UNITS: Dict[str, str] = {
    "return_1y": "%",
    "return_3y_annualized": "%",
    "expense_ratio": "%",
    "volatility_1y": "%",
    "max_drawdown_3y": "%",
    "assets_billions": "$B",
    "return_30d": "%",
    "return_60d": "%",
    "return_90d": "%",
}


def _coerce_percent(value: Optional[float]) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value) * 100 if abs(value) <= 1 else float(value)


def _max_drawdown(prices: pd.Series) -> Optional[float]:
    if prices.empty:
        return None
    running_max = prices.cummax()
    drawdown = (prices / running_max) - 1
    return abs(float(drawdown.min())) * 100


def _annualized_return(prices: pd.Series, trading_days: int = 252) -> Optional[float]:
    if prices.empty or len(prices) < 2:
        return None
    total_return = prices.iloc[-1] / prices.iloc[0]
    years = len(prices) / trading_days
    if years <= 0:
        return None
    return (float(total_return) ** (1 / years) - 1) * 100


def _lookback_return(prices: pd.Series, trading_days: int) -> Optional[float]:
    """Trailing % return over the last `trading_days` bars — same semantics
    as stock_finder_service._pct_return, kept local since funds and stocks
    build their rows from separate history fetches."""
    if prices.empty or len(prices) <= trading_days:
        return None
    start = float(prices.iloc[-trading_days - 1])
    end = float(prices.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100


def _build_fund_row(ticker_symbol: str, fallback_category: str = "Custom") -> Optional[Dict[str, object]]:
    try:
        # Shared cache (services/yfinance_cache.py): dedupes against Stock
        # Finder, Goal Plan, the entry-strategy scanner, etc. pulling the
        # same ticker's history/info within the same 15-minute window.
        history_1y = get_cached_history(ticker_symbol, "1y", auto_adjust=True)
        history_3y = get_cached_history(ticker_symbol, "3y", auto_adjust=True)
        info = get_cached_info(ticker_symbol)

        if history_1y.empty:
            return None

        close_1y = history_1y["Close"]
        close_3y = history_3y["Close"] if not history_3y.empty else close_1y
        daily_returns = close_1y.pct_change().dropna()

        latest_price = float(close_1y.iloc[-1])
        return_1y = ((latest_price / float(close_1y.iloc[0])) - 1) * 100
        volatility_1y = float(daily_returns.std() * np.sqrt(252) * 100) if not daily_returns.empty else None
        return_3y_annualized = _annualized_return(close_3y)
        max_drawdown_3y = _max_drawdown(close_3y)

        return_10d = _lookback_return(close_1y, 10)
        return_30d = _lookback_return(close_1y, 30)
        return_60d = _lookback_return(close_1y, 60)
        return_90d = _lookback_return(close_1y, 90)

        expense_ratio = (
            info.get("annualReportExpenseRatio")
            or info.get("netExpenseRatio")
            or info.get("expenseRatio")
            or info.get("totalExpenseRatio")
        )
        assets = info.get("totalAssets")

        name = info.get("shortName") or info.get("longName") or ticker_symbol
        benchmark = info.get("fundFamily") or info.get("category") or "Yahoo Finance"
        category = info.get("category") or fallback_category

        # yfinance returns this as a Unix timestamp (seconds) when the fund
        # discloses it; not every fund does, so this is often None — shown
        # as "unknown" rather than omitted, so a missing value reads as
        # "the data isn't there" and not as "founded in 1970."
        inception_ts = info.get("fundInceptionDate")
        inception_date = (
            datetime.fromtimestamp(inception_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if inception_ts
            else None
        )

        return {
            "Ticker": ticker_symbol,
            "Fund": name,
            "Benchmark": benchmark,
            "Category": category,
            "Price": latest_price,
            "Expense Ratio %": _coerce_percent(expense_ratio),
            "1Y Return %": return_1y,
            "3Y Annualized %": return_3y_annualized,
            "1Y Volatility %": volatility_1y,
            "3Y Max Drawdown %": max_drawdown_3y,
            "Assets ($B)": (float(assets) / 1_000_000_000) if assets else None,
            "Return 10D %": return_10d,
            "Return 30D %": return_30d,
            "Return 60D %": return_60d,
            "Return 90D %": return_90d,
            "Inception Date": inception_date,
        }
    except Exception:
        return None


@ttl_cache(maxsize=8, ttl_seconds=86400)
def get_index_fund_table() -> pd.DataFrame:
    """
    Builds one row per fund in INDEX_FUND_UNIVERSE, fetched in parallel
    (mirrors services/market_data_service.py's _fetch_closes_parallel
    pattern) — at ~62 funds x 3 yfinance calls each, doing this
    sequentially would be slow and risk Yahoo rate limits the same way a
    large sequential ticker loop already has elsewhere in this app.
    fetch_with_backoff inside _build_fund_row adds pacing/retry on top.

    Cached for 24 hours: fund-level metrics (expense ratio, 1Y/3Y return,
    volatility, inception date) don't meaningfully change intraday, and
    this is one of the more expensive yfinance-touching endpoints in the
    app (~62 funds x 3 calls) — a shorter TTL just re-pays that cost for
    data that looks the same.
    """
    rows: List[Dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = {
            executor.submit(_build_fund_row, fund.ticker, fund.category): fund
            for fund in INDEX_FUND_UNIVERSE
        }
        for future in as_completed(futures):
            fund = futures[future]
            row = future.result()
            if row is not None:
                row["Fund"] = fund.name
                row["Benchmark"] = fund.benchmark
                row["Category"] = fund.category
                rows.append(row)

    order = {fund.ticker: i for i, fund in enumerate(INDEX_FUND_UNIVERSE)}
    rows.sort(key=lambda r: order.get(r["Ticker"], len(order)))
    return pd.DataFrame(rows)


@ttl_cache(maxsize=64, ttl_seconds=86400)
def get_single_fund_table(ticker_symbol: str) -> pd.DataFrame:
    cleaned = ticker_symbol.strip().upper()
    if not cleaned:
        return pd.DataFrame()
    row = _build_fund_row(cleaned)
    return pd.DataFrame([row]) if row is not None else pd.DataFrame()


def _score_series(series: pd.Series, lower_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([0.0] * len(series), index=series.index)

    min_val = numeric.min()
    max_val = numeric.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        base = pd.Series([1.0] * len(series), index=series.index)
    else:
        base = (numeric - min_val) / (max_val - min_val)

    if lower_is_better:
        base = 1 - base

    return base.fillna(base.mean() if not pd.isna(base.mean()) else 0.0)


def rank_index_funds(goal: str, category: str) -> pd.DataFrame:
    df = get_index_fund_table().copy()
    if df.empty:
        return df

    if category != "All":
        df = df[df["Category"] == category].copy()

    if df.empty:
        return df

    df["expense_ratio"] = df["Expense Ratio %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["volatility_1y"] = df["1Y Volatility %"]
    df["max_drawdown_3y"] = df["3Y Max Drawdown %"]
    df["assets_billions"] = df["Assets ($B)"]
    df["return_30d"] = df["Return 30D %"]
    df["return_60d"] = df["Return 60D %"]
    df["return_90d"] = df["Return 90D %"]

    weights = GOAL_WEIGHTS[goal]
    score = pd.Series([0.0] * len(df), index=df.index)

    for metric, weight in weights.items():
        score += _score_series(df[metric], metric in LOWER_IS_BETTER) * weight

    df["Score"] = (score * 100).round(1)
    return df.sort_values(["Score", "1Y Return %", "Assets ($B)"], ascending=[False, False, False]).reset_index(drop=True)


def score_fund_ticker(goal: str, ticker_symbol: str) -> pd.DataFrame:
    df = get_single_fund_table(ticker_symbol).copy()
    if df.empty:
        return df

    df["expense_ratio"] = df["Expense Ratio %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["volatility_1y"] = df["1Y Volatility %"]
    df["max_drawdown_3y"] = df["3Y Max Drawdown %"]
    df["assets_billions"] = df["Assets ($B)"]
    df["return_30d"] = df["Return 30D %"]
    df["return_60d"] = df["Return 60D %"]
    df["return_90d"] = df["Return 90D %"]

    weights = GOAL_WEIGHTS[goal]
    score = pd.Series([0.0] * len(df), index=df.index)
    for metric, weight in weights.items():
        score += _score_series(df[metric], metric in LOWER_IS_BETTER) * weight

    df["Score"] = (score * 100).round(1)
    return df.reset_index(drop=True)
