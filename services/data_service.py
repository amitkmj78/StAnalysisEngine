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
            return None
        return round(float(data["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning("Error fetching latest price for %s: %s", ticker, e)
        return None
