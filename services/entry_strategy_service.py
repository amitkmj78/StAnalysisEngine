from __future__ import annotations

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from services.cache_utils import ttl_cache
from services.index_fund_service import INDEX_FUND_UNIVERSE
from services.screener_service import INDEX_MAP


ENTRY_STOCK_UNIVERSES = {
    "All": sorted({ticker for group in INDEX_MAP.values() for ticker in group}),
    **INDEX_MAP,
}

ENTRY_FUND_UNIVERSES = {
    "All": [fund.ticker for fund in INDEX_FUND_UNIVERSE],
    "US Large Blend": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "US Large Blend"],
    "US Total Market": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "US Total Market"],
    "US Growth": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "US Growth"],
    "US Small Cap": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "US Small Cap"],
    "International": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "International"],
    "Bond": [fund.ticker for fund in INDEX_FUND_UNIVERSE if fund.category == "Bond"],
}


@ttl_cache(maxsize=128, ttl_seconds=3600)
def get_entry_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    cleaned = ticker.strip().upper()
    if not cleaned:
        return pd.DataFrame()
    try:
        return yf.Ticker(cleaned).history(period=period, auto_adjust=True).dropna()
    except Exception:
        return pd.DataFrame()


def _signal_rank(signal: str) -> int:
    order = {
        "Buy Now": 5,
        "Buy on Pullback": 4,
        "Breakout Entry": 3,
        "Watch for Reversal": 2,
        "Wait for Pullback": 1,
        "Wait": 0,
    }
    return order.get(signal, 0)


def build_entry_plan(ticker: str) -> dict | None:
    hist = get_entry_history(ticker, "1y")
    if hist.empty or len(hist) < 80:
        return None

    df = hist.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
    atr_indicator = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)

    current_price = float(close.iloc[-1])
    sma20 = float(df["SMA20"].iloc[-1]) if pd.notna(df["SMA20"].iloc[-1]) else None
    sma50 = float(df["SMA50"].iloc[-1]) if pd.notna(df["SMA50"].iloc[-1]) else None
    sma200 = float(df["SMA200"].iloc[-1]) if pd.notna(df["SMA200"].iloc[-1]) else None
    rsi = float(rsi_indicator.rsi().iloc[-1]) if pd.notna(rsi_indicator.rsi().iloc[-1]) else None
    atr = float(atr_indicator.average_true_range().iloc[-1]) if pd.notna(atr_indicator.average_true_range().iloc[-1]) else None
    macd = float(macd_indicator.macd().iloc[-1]) if pd.notna(macd_indicator.macd().iloc[-1]) else None
    macd_signal = float(macd_indicator.macd_signal().iloc[-1]) if pd.notna(macd_indicator.macd_signal().iloc[-1]) else None

    support_20 = float(low.tail(20).min())
    support_60 = float(low.tail(60).min())
    resistance_20 = float(high.tail(20).max())
    resistance_60 = float(high.tail(60).max())
    avg_volume_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else None
    latest_volume = float(volume.iloc[-1]) if not volume.empty else None

    trend_up = bool(sma20 and sma50 and current_price > sma20 > sma50)
    long_term_up = bool(sma50 and sma200 and sma50 > sma200)
    overextended = bool(rsi and rsi >= 70)
    oversold = bool(rsi and rsi <= 35)
    bullish_momentum = bool(macd is not None and macd_signal is not None and macd > macd_signal)
    near_support = current_price <= support_20 * 1.03
    near_breakout = current_price >= resistance_20 * 0.98

    if trend_up and bullish_momentum and not overextended and current_price <= resistance_20:
        signal = "Buy Now"
        summary = "Trend and momentum are constructive without looking overheated."
    elif trend_up and overextended:
        signal = "Wait for Pullback"
        summary = "The trend is healthy, but price looks stretched. A calmer entry is safer."
    elif long_term_up and near_support:
        signal = "Buy on Pullback"
        summary = "The broader structure is still solid and price is near support."
    elif near_breakout and bullish_momentum:
        signal = "Breakout Entry"
        summary = "Price is pressing resistance with supportive momentum."
    elif oversold:
        signal = "Watch for Reversal"
        summary = "The stock is washed out. Wait for confirmation before entering."
    else:
        signal = "Wait"
        summary = "The setup is mixed right now. Waiting for either support or stronger momentum is cleaner."

    ideal_entry_low = min(support_20 * 1.00, current_price)
    ideal_entry_high = min(max(support_20 * 1.03, current_price * 0.995), resistance_20)
    breakout_entry = resistance_20 * 1.01
    stop_loss = support_20 - (atr or current_price * 0.03)
    first_target = current_price + ((current_price - stop_loss) * 2)

    entry_score = 0.0
    entry_score += _signal_rank(signal) * 18
    if rsi is not None:
        entry_score += max(0.0, 20 - abs(rsi - 52))
    if bullish_momentum:
        entry_score += 14
    if trend_up:
        entry_score += 14
    if long_term_up:
        entry_score += 10
    if near_support:
        entry_score += 12
    elif near_breakout:
        entry_score += 8
    if avg_volume_20 and latest_volume:
        volume_ratio = latest_volume / avg_volume_20
        entry_score += max(0.0, min(12.0, (volume_ratio - 1.0) * 20))
    entry_score = round(min(100.0, max(0.0, entry_score)), 1)

    return {
        "ticker": ticker.strip().upper(),
        "history": df,
        "current_price": current_price,
        "signal": signal,
        "summary": summary,
        "rsi": rsi,
        "atr": atr,
        "macd": macd,
        "macd_signal": macd_signal,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "support_20": support_20,
        "support_60": support_60,
        "resistance_20": resistance_20,
        "resistance_60": resistance_60,
        "ideal_entry_low": ideal_entry_low,
        "ideal_entry_high": ideal_entry_high,
        "breakout_entry": breakout_entry,
        "stop_loss": stop_loss,
        "first_target": first_target,
        "avg_volume_20": avg_volume_20,
        "latest_volume": latest_volume,
        "trend_up": trend_up,
        "long_term_up": long_term_up,
        "entry_score": entry_score,
    }


def scan_best_entries(asset_type: str, universe_key: str) -> pd.DataFrame:
    if asset_type == "Fund":
        tickers = ENTRY_FUND_UNIVERSES.get(universe_key, [])
    else:
        tickers = ENTRY_STOCK_UNIVERSES.get(universe_key, [])

    rows: list[dict] = []
    for ticker in tickers:
        plan = build_entry_plan(ticker)
        if not plan:
            continue
        rows.append(
            {
                "Ticker": plan["ticker"],
                "Signal": plan["signal"],
                "Entry Score": plan["entry_score"],
                "Current Price": plan["current_price"],
                "Entry Low": plan["ideal_entry_low"],
                "Entry High": plan["ideal_entry_high"],
                "Breakout Entry": plan["breakout_entry"],
                "Stop Loss": plan["stop_loss"],
                "First Target": plan["first_target"],
                "RSI": plan["rsi"],
                "Support 20D": plan["support_20"],
                "Resistance 20D": plan["resistance_20"],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Signal Rank"] = df["Signal"].map(_signal_rank).fillna(0)
    df = df.sort_values(["Entry Score", "Signal Rank"], ascending=[False, False]).reset_index(drop=True)
    return df.drop(columns=["Signal Rank"])
