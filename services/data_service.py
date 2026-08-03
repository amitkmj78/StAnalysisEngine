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


@ttl_cache(maxsize=256, ttl_seconds=60)
def get_latest_price(ticker: str):
    """Get the latest closing price for the given ticker."""
    if not ticker:
        return None
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


@ttl_cache(maxsize=256, ttl_seconds=60)
def get_extended_hours_price(ticker: str):
    """
    Pre/post-market price, when the market is actually in one of those states.
    yfinance's regular history/fast_info only reflect the regular session —
    a stock can move sharply after hours (earnings, news) and that's invisible
    there, which reads as "wrong" even though the regular-session number is
    correct for what it is. Returns None outside pre/post market hours.
    """
    if not ticker:
        return None
    try:
        info = yf.Ticker(ticker).info
        state = info.get("marketState")
        if state == "POST":
            price = info.get("postMarketPrice")
            change_pct = info.get("postMarketChangePercent")
        elif state == "PRE":
            price = info.get("preMarketPrice")
            change_pct = info.get("preMarketChangePercent")
        else:
            return None
        if price is None:
            return None
        return {
            "state": state,
            "price": round(float(price), 2),
            "change_pct": round(float(change_pct), 2) if change_pct is not None else None,
        }
    except Exception as e:
        logger.warning("Error fetching extended-hours price for %s: %s", ticker, e)
        return None
