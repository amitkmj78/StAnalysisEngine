from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from services.cache_utils import ttl_cache
from services.screener_service import INDEX_MAP


STOCK_UNIVERSES: Dict[str, List[str]] = {
    "All": sorted({ticker for group in INDEX_MAP.values() for ticker in group}),
    **INDEX_MAP,
}


LOWER_IS_BETTER = {"volatility_6m", "max_drawdown_1y", "forward_pe"}

METRIC_LABELS: Dict[str, str] = {
    "return_1m": "1-Month Return",
    "return_3m": "3-Month Return",
    "return_6m": "6-Month Return",
    "return_1y": "1-Year Return",
    "return_3y_annualized": "3-Year Annualized Return",
    "rsi_balance": "RSI Balance",
    "macd_signal_strength": "MACD Signal Strength",
    "volume_strength": "Volume Strength",
    "volatility_6m": "6-Month Volatility",
    "max_drawdown_1y": "1-Year Max Drawdown",
    "forward_pe": "Forward P/E",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
}

METRIC_UNITS: Dict[str, str] = {
    "return_1m": "%",
    "return_3m": "%",
    "return_6m": "%",
    "return_1y": "%",
    "return_3y_annualized": "%",
    "rsi_balance": "pts",
    "macd_signal_strength": "pts",
    "volume_strength": "%",
    "volatility_6m": "%",
    "max_drawdown_1y": "%",
    "forward_pe": "x",
    "revenue_growth": "%",
    "earnings_growth": "%",
}


GOAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Short Term": {
        "return_1m": 0.25,
        "return_3m": 0.30,
        "rsi_balance": 0.15,
        "macd_signal_strength": 0.15,
        "volume_strength": 0.10,
        "volatility_6m": 0.05,
    },
    "Long Term": {
        "return_1y": 0.28,
        "return_6m": 0.12,
        "return_3y_annualized": 0.20,
        "revenue_growth": 0.12,
        "earnings_growth": 0.10,
        "forward_pe": 0.08,
        "max_drawdown_1y": 0.10,
    },
}


def _pct_return(close: pd.Series, lookback: int) -> float | None:
    if close.empty or len(close) <= lookback:
        return None
    start = float(close.iloc[-lookback - 1])
    end = float(close.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100


def _annualized_return(close: pd.Series, trading_days: int = 252) -> float | None:
    if close.empty or len(close) < 2:
        return None
    years = len(close) / trading_days
    if years <= 0:
        return None
    total_return = float(close.iloc[-1]) / float(close.iloc[0])
    return (total_return ** (1 / years) - 1.0) * 100


def _max_drawdown(close: pd.Series) -> float | None:
    if close.empty:
        return None
    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    return abs(float(drawdown.min())) * 100


def _safe_percent(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value * 100 if abs(value) <= 1 else value


def _score_series(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([0.0] * len(series), index=series.index)

    min_val = numeric.min()
    max_val = numeric.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        scaled = pd.Series([1.0] * len(series), index=series.index)
    else:
        scaled = (numeric - min_val) / (max_val - min_val)

    if lower_is_better:
        scaled = 1 - scaled

    mean_val = scaled.mean()
    return scaled.fillna(mean_val if not pd.isna(mean_val) else 0.0)


def _rsi_balance_score(rsi: float | None) -> float | None:
    if rsi is None or pd.isna(rsi):
        return None
    # Prefer not-overheated but still strong momentum.
    return max(0.0, 100 - abs(float(rsi) - 55) * 3)


def _build_stock_row(ticker_symbol: str) -> dict | None:
    try:
        yf_ticker = yf.Ticker(ticker_symbol)
        hist = yf_ticker.history(period="3y", auto_adjust=True).dropna()
        info = yf_ticker.info or {}

        if hist.empty or len(hist) < 70:
            return None

        close = hist["Close"]
        volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(dtype=float)
        latest_price = float(close.iloc[-1])

        return_1m = _pct_return(close, 21)
        return_3m = _pct_return(close, 63)
        return_6m = _pct_return(close, 126)
        return_1y = _pct_return(close, 252)
        return_3y_annualized = _annualized_return(close)

        # Literal trading-day trailing windows for the Top Performers
        # leaderboard — deliberately separate from the 1M/3M/6M/1Y columns
        # above (21/63/126/252-day approximations used by the composite
        # scoring system) so the two features can't drift into confusing
        # near-duplicates of each other.
        return_10d = _pct_return(close, 10)
        return_30d = _pct_return(close, 30)
        return_60d = _pct_return(close, 60)
        return_90d = _pct_return(close, 90)
        max_drawdown_1y = _max_drawdown(close.tail(252))

        daily_returns_6m = close.tail(126).pct_change().dropna()
        volatility_6m = (
            float(daily_returns_6m.std() * np.sqrt(252) * 100)
            if not daily_returns_6m.empty
            else None
        )

        try:
            rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        except Exception:
            rsi = None

        try:
            macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            macd_value = float(macd_indicator.macd().iloc[-1])
            macd_signal = float(macd_indicator.macd_signal().iloc[-1])
            macd_signal_strength = (macd_value - macd_signal) * 100
        except Exception:
            macd_signal_strength = None

        try:
            recent_volume = float(volume.iloc[-1])
            avg_volume = float(volume.tail(20).mean())
            volume_strength = ((recent_volume / avg_volume) - 1.0) * 100 if avg_volume else None
        except Exception:
            volume_strength = None

        return {
            "Ticker": ticker_symbol,
            "Name": info.get("shortName") or info.get("longName") or ticker_symbol,
            "Sector": info.get("sector") or "Unknown",
            "Price": latest_price,
            "Market Cap ($B)": (
                float(info.get("marketCap")) / 1_000_000_000
                if info.get("marketCap")
                else None
            ),
            "Forward PE": info.get("forwardPE"),
            "Revenue Growth %": _safe_percent(info.get("revenueGrowth")),
            "Earnings Growth %": _safe_percent(info.get("earningsGrowth")),
            "1M Return %": return_1m,
            "3M Return %": return_3m,
            "6M Return %": return_6m,
            "1Y Return %": return_1y,
            "3Y Annualized %": return_3y_annualized,
            "Return 10D %": return_10d,
            "Return 30D %": return_30d,
            "Return 60D %": return_60d,
            "Return 90D %": return_90d,
            "RSI": rsi,
            "RSI Balance": _rsi_balance_score(rsi),
            "MACD Strength": macd_signal_strength,
            "Volume Strength %": volume_strength,
            "6M Volatility %": volatility_6m,
            "1Y Max Drawdown %": max_drawdown_1y,
        }
    except Exception:
        return None


@ttl_cache(maxsize=64, ttl_seconds=3600)
def get_stock_finder_table(universe_key: str) -> pd.DataFrame:
    tickers = STOCK_UNIVERSES.get(universe_key, [])
    rows: List[dict] = []

    for ticker in tickers:
        row = _build_stock_row(ticker)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


@ttl_cache(maxsize=64, ttl_seconds=3600)
def get_single_stock_table(ticker_symbol: str) -> pd.DataFrame:
    cleaned = ticker_symbol.strip().upper()
    if not cleaned:
        return pd.DataFrame()
    row = _build_stock_row(cleaned)
    return pd.DataFrame([row]) if row is not None else pd.DataFrame()


def rank_stocks(goal: str, universe_key: str) -> pd.DataFrame:
    df = get_stock_finder_table(universe_key).copy()
    if df.empty:
        return df

    df["return_1m"] = df["1M Return %"]
    df["return_3m"] = df["3M Return %"]
    df["return_6m"] = df["6M Return %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["rsi_balance"] = df["RSI Balance"]
    df["macd_signal_strength"] = df["MACD Strength"]
    df["volume_strength"] = df["Volume Strength %"]
    df["volatility_6m"] = df["6M Volatility %"]
    df["max_drawdown_1y"] = df["1Y Max Drawdown %"]
    df["forward_pe"] = df["Forward PE"]
    df["revenue_growth"] = df["Revenue Growth %"]
    df["earnings_growth"] = df["Earnings Growth %"]

    weights = GOAL_WEIGHTS[goal]
    total_score = pd.Series([0.0] * len(df), index=df.index)

    for metric, weight in weights.items():
        total_score += _score_series(df[metric], lower_is_better=metric in LOWER_IS_BETTER) * weight

    df["Score"] = (total_score * 100).round(1)

    secondary_sort = "3M Return %" if goal == "Short Term" else "1Y Return %"
    return (
        df.sort_values(["Score", secondary_sort, "Market Cap ($B)"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def score_stock_ticker(goal: str, ticker_symbol: str) -> pd.DataFrame:
    df = get_single_stock_table(ticker_symbol).copy()
    if df.empty:
        return df

    df["return_1m"] = df["1M Return %"]
    df["return_3m"] = df["3M Return %"]
    df["return_6m"] = df["6M Return %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["rsi_balance"] = df["RSI Balance"]
    df["macd_signal_strength"] = df["MACD Strength"]
    df["volume_strength"] = df["Volume Strength %"]
    df["volatility_6m"] = df["6M Volatility %"]
    df["max_drawdown_1y"] = df["1Y Max Drawdown %"]
    df["forward_pe"] = df["Forward PE"]
    df["revenue_growth"] = df["Revenue Growth %"]
    df["earnings_growth"] = df["Earnings Growth %"]

    weights = GOAL_WEIGHTS[goal]
    total_score = pd.Series([0.0] * len(df), index=df.index)
    for metric, weight in weights.items():
        total_score += _score_series(df[metric], lower_is_better=metric in LOWER_IS_BETTER) * weight

    df["Score"] = (total_score * 100).round(1)
    return df.reset_index(drop=True)
