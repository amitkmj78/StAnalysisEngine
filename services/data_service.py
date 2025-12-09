from typing import Dict

import streamlit as st
import yfinance as yf
import pandas as pd


# Timeframe labels → yfinance period codes
TIMEFRAME_MAPPING: Dict[str, str] = {
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
}


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Fetch historical stock data for a given ticker and period."""
    if not ticker:
        return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period).dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
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
        st.error(f"Error fetching latest price for {ticker}: {e}")
        return None
