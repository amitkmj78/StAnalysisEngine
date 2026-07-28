from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from services.cache_utils import ttl_cache


@dataclass(frozen=True)
class IndexFundCandidate:
    ticker: str
    name: str
    benchmark: str
    category: str


INDEX_FUND_UNIVERSE: List[IndexFundCandidate] = [
    IndexFundCandidate("VOO", "Vanguard S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("IVV", "iShares Core S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("SPLG", "SPDR Portfolio S&P 500 ETF", "S&P 500", "US Large Blend"),
    IndexFundCandidate("SPY", "SPDR S&P 500 ETF Trust", "S&P 500", "US Large Blend"),
    IndexFundCandidate("VTI", "Vanguard Total Stock Market ETF", "CRSP US Total Market", "US Total Market"),
    IndexFundCandidate("ITOT", "iShares Core S&P Total US Stock Market ETF", "S&P Total US Stock Market", "US Total Market"),
    IndexFundCandidate("SCHB", "Schwab US Broad Market ETF", "Dow Jones US Broad Stock Market", "US Total Market"),
    IndexFundCandidate("QQQ", "Invesco QQQ Trust", "Nasdaq-100", "US Growth"),
    IndexFundCandidate("VUG", "Vanguard Growth ETF", "CRSP US Large Cap Growth", "US Growth"),
    IndexFundCandidate("IWM", "iShares Russell 2000 ETF", "Russell 2000", "US Small Cap"),
    IndexFundCandidate("VXUS", "Vanguard Total International Stock ETF", "FTSE Global All Cap ex US", "International"),
    IndexFundCandidate("IXUS", "iShares Core MSCI Total International Stock ETF", "MSCI ACWI ex USA IMI", "International"),
    IndexFundCandidate("BND", "Vanguard Total Bond Market ETF", "Bloomberg US Aggregate Float Adjusted", "Bond"),
    IndexFundCandidate("AGG", "iShares Core US Aggregate Bond ETF", "Bloomberg US Aggregate Bond", "Bond"),
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
    "Best Growth": {
        "return_1y": 0.50,
        "return_3y_annualized": 0.35,
        "expense_ratio": 0.05,
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
}

METRIC_UNITS: Dict[str, str] = {
    "return_1y": "%",
    "return_3y_annualized": "%",
    "expense_ratio": "%",
    "volatility_1y": "%",
    "max_drawdown_3y": "%",
    "assets_billions": "$B",
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


def _build_fund_row(ticker_symbol: str, fallback_category: str = "Custom") -> Optional[Dict[str, object]]:
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_1y = ticker.history(period="1y", auto_adjust=True).dropna()
        history_3y = ticker.history(period="3y", auto_adjust=True).dropna()
        info = ticker.info or {}

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
        }
    except Exception:
        return None


@ttl_cache(maxsize=8, ttl_seconds=3600)
def get_index_fund_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for fund in INDEX_FUND_UNIVERSE:
        row = _build_fund_row(fund.ticker, fallback_category=fund.category)
        if row is not None:
            row["Fund"] = fund.name
            row["Benchmark"] = fund.benchmark
            row["Category"] = fund.category
            rows.append(row)

    return pd.DataFrame(rows)


@ttl_cache(maxsize=64, ttl_seconds=3600)
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

    weights = GOAL_WEIGHTS[goal]
    score = pd.Series([0.0] * len(df), index=df.index)
    for metric, weight in weights.items():
        score += _score_series(df[metric], metric in LOWER_IS_BETTER) * weight

    df["Score"] = (score * 100).round(1)
    return df.reset_index(drop=True)
