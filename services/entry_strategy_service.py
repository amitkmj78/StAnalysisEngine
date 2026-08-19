from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import ta
import yfinance as yf

from services.cache_utils import ttl_cache
from services.index_fund_service import INDEX_FUND_UNIVERSE
from services.screener_service import INDEX_MAP
from services.stock_finder_service import SP500_UNIVERSE_NAME, fetch_sp500_tickers

MAX_PARALLEL_FETCHES = 10

# "All" and SP500_UNIVERSE_NAME resolve their ticker lists lazily via
# _entry_stock_tickers (a live, 24h-cached Wikipedia fetch shared with
# stock_finder_service.py) rather than at import time — a blocked/slow
# network call must never delay app startup. These keys exist here as
# empty placeholders purely so /universes listing and the router's
# `universe in ENTRY_STOCK_UNIVERSES` validation keep working.
ENTRY_STOCK_UNIVERSES: dict[str, list[str]] = {
    "All": [],
    SP500_UNIVERSE_NAME: [],
    **INDEX_MAP,
}


def _entry_stock_tickers(universe_key: str) -> list[str]:
    if universe_key == SP500_UNIVERSE_NAME:
        return fetch_sp500_tickers()
    if universe_key == "All":
        return sorted({t for group in INDEX_MAP.values() for t in group} | set(fetch_sp500_tickers()))
    return ENTRY_STOCK_UNIVERSES.get(universe_key, [])


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

    # Each flat bonus below only counts as *extra* evidence beyond what
    # the signal label already guarantees by its own definition (see the
    # if/elif chain above) — e.g. "Buy Now" requires trend_up AND
    # bullish_momentum to be assigned at all, so re-awarding those same
    # +14/+14 points there would just double-count the same evidence.
    # Before this guard, every "Buy Now"/"Buy on Pullback"/"Breakout
    # Entry" stock started from an inflated baseline that nearly always
    # saturated the 100 cap regardless of how much stronger one setup
    # genuinely was than another (visible at S&P 500 scale: a third of
    # scanned stocks tied at exactly 100.0). The guards below match the
    # signal branches 1:1 so only real additional strength moves the
    # score within each bucket.
    if bullish_momentum and signal not in ("Buy Now", "Breakout Entry"):
        entry_score += 14
    if trend_up and signal not in ("Buy Now", "Wait for Pullback"):
        entry_score += 14
    if long_term_up and signal != "Buy on Pullback":
        entry_score += 10
    if near_support and signal != "Buy on Pullback":
        entry_score += 12
    elif near_breakout and signal != "Breakout Entry":
        entry_score += 8
    if avg_volume_20 and latest_volume:
        volume_ratio = latest_volume / avg_volume_20
        entry_score += max(0.0, min(12.0, (volume_ratio - 1.0) * 20))
    entry_score = max(0.0, entry_score)
    # Kept alongside the capped/rounded display score purely as a scan-
    # level tiebreaker (scan_best_entries) — two stocks can genuinely
    # differ in underlying strength while both showing "100" once
    # capped; without this, ties fell back to whatever order the
    # parallel scan's network calls happened to complete in (looked
    # arbitrary/alphabetical-ish rather than ranked).
    raw_entry_score = entry_score
    entry_score = round(min(100.0, entry_score), 1)

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
        "raw_entry_score": raw_entry_score,
    }


def _entry_row(ticker: str) -> dict | None:
    plan = build_entry_plan(ticker)
    if not plan:
        return None
    return {
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
        "_raw_entry_score": plan["raw_entry_score"],
    }


def scan_best_entries(asset_type: str, universe_key: str) -> pd.DataFrame:
    if asset_type == "Fund":
        tickers = ENTRY_FUND_UNIVERSES.get(universe_key, [])
    else:
        tickers = _entry_stock_tickers(universe_key)

    # build_entry_plan is a couple of independent, I/O-bound yfinance
    # calls per ticker — parallelize so the S&P 500 (up to 500 tickers)
    # is tractable within the endpoint's request lifetime, same pattern
    # as stock_finder_service.get_stock_finder_table. Order doesn't
    # matter here since results are sorted afterward.
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = [executor.submit(_entry_row, ticker) for ticker in tickers]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Signal Rank"] = df["Signal"].map(_signal_rank).fillna(0)
    # _raw_entry_score (pre-cap, more decimal precision) breaks ties
    # between stocks that both display "100" once capped — without it,
    # ties fell back to whatever order the parallel scan's network calls
    # happened to complete in, which looked arbitrary (and, since
    # fetch_sp500_tickers() returns tickers alphabetically and the
    # earliest-submitted threads tend to finish first, misleadingly
    # close to alphabetical rather than actually ranked). Ticker is a
    # final deterministic tiebreak for the vanishingly unlikely case two
    # stocks also match on raw score.
    df = df.sort_values(
        ["Entry Score", "Signal Rank", "_raw_entry_score", "Ticker"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return df.drop(columns=["Signal Rank", "_raw_entry_score"])
