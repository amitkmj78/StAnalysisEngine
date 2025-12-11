# services/screener_service.py

from typing import Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
import ta

from services.analysis_service import get_analyst_rating_summary

# ============================
#  INDEX UNIVERSES (ABC)
# ============================
# A = US MAJORS, B = GLOBAL, C = CRYPTO

INDEX_MAP: Dict[str, List[str]] = {
    # ---------- A: US Major Equity ----------
    "US - Mega Cap (SPY sample)": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META",
        "BRK-B", "JPM", "UNH", "XOM", "AVGO", "LLY", "V", "JNJ",
        "TSLA", "WMT", "PG", "MA", "HD"
    ],
    "US - Tech Growth (QQQ sample)": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "ADBE", "CRM", "NFLX", "AMD", "INTC"
    ],
    # You can add your other custom universes here...
}


# -----------------------------
# Helpers
# -----------------------------
def _safe_pct_return(start: float, end: float) -> float:
    if start is None or end is None:
        return np.nan
    if np.isnan(start) or np.isnan(end) or start == 0:
        return np.nan
    return (end / start - 1.0) * 100.0


def _compute_technicals(hist: pd.DataFrame) -> dict:
    """Compute RSI, MACD and volume strength from OHLCV history."""
    if hist is None or hist.empty:
        return {"rsi": np.nan, "macd": np.nan, "macd_signal": np.nan, "volume_strength": np.nan}

    close = hist["Close"]
    vol = hist["Volume"]

    # RSI (14)
    try:
        rsi_ind = ta.momentum.RSIIndicator(close, window=14)
        rsi = float(rsi_ind.rsi().iloc[-1])
    except Exception:
        rsi = np.nan

    # MACD (26, 12, 9)
    try:
        macd_ind = ta.trend.MACD(
            close,
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )
        macd = float(macd_ind.macd().iloc[-1])
        macd_signal = float(macd_ind.macd_signal().iloc[-1])
    except Exception:
        macd, macd_signal = np.nan, np.nan

    # Volume strength – latest vs 20-day avg → scaled ~0–10
    try:
        if len(vol.dropna()) >= 20:
            v_last = float(vol.iloc[-1])
            v_avg20 = float(vol.tail(20).mean())
            raw = (v_last / v_avg20) - 1.0
            volume_strength = max(0.0, min(10.0, raw * 10.0))  # e.g. +100% vol → ~10
        else:
            volume_strength = np.nan
    except Exception:
        volume_strength = np.nan

    return {
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "volume_strength": volume_strength,
    }


def _compute_bullish_score(
    ticker: str,
    price: float,
    ret_1y: float,
    ret_6m: float,
    ret_3m: float,
    rsi: float,
    macd: float,
    macd_signal: float,
    volume_strength: float,
    analyst_summary: dict | None = None,
) -> dict:
    """
    Bullish Score v2.0
    - 1Y / 6M / 3M momentum
    - RSI
    - MACD
    - Volume strength
    - Optional analyst tilt
    """
    score = 0.0

    # 1-Year Momentum (up to 30)
    if pd.notna(ret_1y):
        score += min(30.0, max(0.0, ret_1y / 4.0))

    # 6-Month Momentum (up to 20)
    if pd.notna(ret_6m):
        score += min(20.0, max(0.0, ret_6m / 3.0))

    # 3-Month Momentum (up to 15)
    if pd.notna(ret_3m):
        score += min(15.0, max(0.0, ret_3m / 2.0))

    # RSI – 15
    if pd.notna(rsi):
        if rsi < 30:
            score += 15.0  # oversold → bullish
        elif 30 <= rsi <= 60:
            score += 10.0  # healthy zone
        elif 60 < rsi < 70:
            score += 5.0   # getting hot
        # >70 → no bonus

    # MACD – 10
    if pd.notna(macd) and pd.notna(macd_signal) and macd > macd_signal:
        score += 10.0

    # Volume – 10
    if pd.notna(volume_strength):
        score += min(10.0, max(0.0, volume_strength))

    # Optional: tiny tilt from analysts (if available)
    analyst_consensus = None
    analyst_buy_pct = None
    if analyst_summary:
        # These keys depend on your implementation of get_analyst_rating_summary()
        analyst_consensus = (
            analyst_summary.get("consensus")
            or analyst_summary.get("rating")
            or analyst_summary.get("overall_rating")
        )
        analyst_buy_pct = analyst_summary.get("buy_percent") or analyst_summary.get("buy_%")

        if analyst_consensus:
            c = analyst_consensus.lower()
            if "strong buy" in c:
                score += 5.0
            elif "buy" in c:
                score += 3.0
            elif "hold" in c:
                score += 1.0
            elif "sell" in c:
                score -= 5.0

    score = round(min(100.0, max(0.0, score)), 2)

    if score >= 75:
        rating = "STRONG BUY 🟢"
    elif score >= 60:
        rating = "BUY 🟩"
    elif score >= 40:
        rating = "WATCH 🟡"
    else:
        rating = "SELL 🔴"

    return {
        "Ticker": ticker,
        "Price": round(price, 2) if pd.notna(price) else None,
        "Return_1Y_%": round(ret_1y, 2) if pd.notna(ret_1y) else None,
        "Return_6M_%": round(ret_6m, 2) if pd.notna(ret_6m) else None,
        "Return_3M_%": round(ret_3m, 2) if pd.notna(ret_3m) else None,
        "RSI": round(rsi, 2) if pd.notna(rsi) else None,
        "Volume_Strength": round(volume_strength, 2) if pd.notna(volume_strength) else None,
        "Score": score,
        "Rating": rating,
        "Analyst_Consensus": analyst_consensus,
        "Analyst_Buy_%": analyst_buy_pct,
    }


# -----------------------------
# Main Screener Entry Point
# -----------------------------
def get_top_bullish_stocks(
    index_key: str,
    period: str = "1y",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Returns a DataFrame of top bullish tickers for a given universe.

    period: "3mo", "6mo", "1y" – used only to decide which return column
            is primary for sorting; all three are still computed.
    """
    tickers = INDEX_MAP.get(index_key, [])
    if not tickers:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Price",
                "Return_1Y_%",
                "Return_6M_%",
                "Return_3M_%",
                "RSI",
                "Volume_Strength",
                "Score",
                "Rating",
                "Analyst_Consensus",
                "Analyst_Buy_%",
            ]
        )

    rows: list[dict] = []

    for tk in tickers:
        try:
            # Pull up to 1 year of data (enough for all windows)
            hist = yf.Ticker(tk).history(period="1y", auto_adjust=True)
            if hist.empty:
                continue

            close = hist["Close"]
            price = float(close.iloc[-1])

            # Compute returns
            ret_1y = _safe_pct_return(float(close.iloc[0]), price)

            if len(close) >= 126:  # ~6M (21*6)
                ret_6m = _safe_pct_return(float(close.iloc[-126]), price)
            else:
                ret_6m = np.nan

            if len(close) >= 63:   # ~3M (21*3)
                ret_3m = _safe_pct_return(float(close.iloc[-63]), price)
            else:
                ret_3m = np.nan

            tech = _compute_technicals(hist)

            # Optional analyst summary (if implemented / available)
            try:
                analyst_summary = get_analyst_rating_summary(tk) or {}
            except Exception:
                analyst_summary = {}

            score_row = _compute_bullish_score(
                ticker=tk,
                price=price,
                ret_1y=ret_1y,
                ret_6m=ret_6m,
                ret_3m=ret_3m,
                rsi=tech["rsi"],
                macd=tech["macd"],
                macd_signal=tech["macd_signal"],
                volume_strength=tech["volume_strength"],
                analyst_summary=analyst_summary,
            )
            rows.append(score_row)

        except Exception:
            # Skip bad tickers quietly; you can add logging here
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Price",
                "Return_1Y_%",
                "Return_6M_%",
                "Return_3M_%",
                "RSI",
                "Volume_Strength",
                "Score",
                "Rating",
                "Analyst_Consensus",
                "Analyst_Buy_%",
            ]
        )

    df = pd.DataFrame(rows)

    # Choose primary return based on UI period
    period_col_map = {
        "3mo": "Return_3M_%",
        "6mo": "Return_6M_%",
        "1y": "Return_1Y_%",
    }
    primary_col = period_col_map.get(period, "Return_1Y_%")
    if primary_col not in df.columns:
        primary_col = "Return_1Y_%"

    # Sort: first by Score desc, then primary return desc
    df = df.sort_values(
        by=["Score", primary_col],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Trim
    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    return df
