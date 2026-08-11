import logging
from typing import Dict

import yfinance as yf
import pandas as pd

from .cache_utils import ttl_cache

logger = logging.getLogger(__name__)


# Timeframe labels → yfinance period codes
TIMEFRAME_MAPPING: Dict[str, str] = {
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
}


@ttl_cache(maxsize=256, ttl_seconds=300)
def get_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch historical stock data for a given ticker and period."""
    if not ticker:
        return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period).dropna()
        return data
    except Exception as e:
        logger.warning("Error fetching data for %s: %s", ticker, e)
        return pd.DataFrame()


@ttl_cache(maxsize=128, ttl_seconds=3600)
def get_adjusted_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Split/dividend-adjusted daily OHLC history (FR-02 of the Safe Baseline
    Price Band spec). Explicitly requests auto_adjust=True rather than
    relying on yfinance's default, since get_stock_data above doesn't
    assert it either way and baseline math must never mix adjusted and
    unadjusted prices. Cached for 1hr, matching get_entry_history's
    pattern — this is historical (non-live) data.
    """
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        return pd.DataFrame()
    try:
        return yf.Ticker(cleaned).history(period=period, auto_adjust=True).dropna()
    except Exception as e:
        logger.warning("Error fetching adjusted history for %s: %s", ticker, e)
        return pd.DataFrame()


@ttl_cache(maxsize=256, ttl_seconds=2)
def get_latest_price(ticker: str):
    """
    Get the latest live price for the given ticker.

    Uses yfinance's lightweight `fast_info` quote (a single quote lookup,
    not a full history fetch) so this stays cheap enough to genuinely poll
    every second or two from the UI — `get_stock_data` is itself cached for
    5 minutes, so routing through it here would make "live" price polling
    silently return the same stale number for minutes at a time.
    """
    if not ticker:
        return None
    try:
        price = yf.Ticker(ticker).fast_info.get("lastPrice")
        if price is not None:
            return round(float(price), 2)
    except Exception as e:
        logger.warning("Error fetching fast_info for %s: %s", ticker, e)

    # Fallback for tickers fast_info can't quote — same as before.
    try:
        data = get_stock_data(ticker, "1d")
        if data.empty:
            # Mutual funds (e.g. FXAIX) often have no "1d" bar since they
            # price once at end of day rather than trading intraday — fall
            # back to a wider window and take the most recent close instead
            # of treating them as unpriceable.
            data = get_stock_data(ticker, "5d")
        if data.empty:
            return None
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning("Error fetching latest price for %s: %s", ticker, e)
        return None


@ttl_cache(maxsize=256, ttl_seconds=10)
def get_extended_hours_price(ticker: str):
    """
    Pre/post-market price, when the market is actually in one of those states.
    yfinance's regular history/fast_info only reflect the regular session —
    a stock can move sharply after hours (earnings, news) and that's invisible
    there, which reads as "wrong" even though the regular-session number is
    correct for what it is. Returns None outside pre/post market hours.

    Cached shorter than before (10s, not 60s) to keep pace with the price
    badge's 1s polling, but longer than get_latest_price's 2s since this
    goes through the heavier `.info` full-quote call, not fast_info.
    """
    if not ticker:
        return None
    try:
        info = yf.Ticker(ticker).info
        state = info.get("marketState") or ""
        # Yahoo isn't just "PRE"/"POST" — it also returns "POSTPOST" (after
        # the post-market session itself has quieted down, still before the
        # next pre-market) and presumably "PREPRE". Matching the exact
        # string meant this silently returned nothing during POSTPOST even
        # though postMarketPrice was populated. Normalize to PRE/POST so the
        # frontend's exact-match check still works.
        if state.startswith("POST"):
            price = info.get("postMarketPrice")
            change_pct = info.get("postMarketChangePercent")
            normalized_state = "POST"
        elif state.startswith("PRE"):
            price = info.get("preMarketPrice")
            change_pct = info.get("preMarketChangePercent")
            normalized_state = "PRE"
        else:
            return None
        if price is None:
            return None
        return {
            "state": normalized_state,
            "price": round(float(price), 2),
            "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
        }
    except Exception as e:
        logger.warning("Error fetching extended-hours price for %s: %s", ticker, e)
        return None
