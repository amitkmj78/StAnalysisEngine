import yfinance as yf
import pandas as pd
from services.data_service import get_stock_data


INDEX_MAP = {
    # 🇺🇸 US Markets
    "S&P 500 (US Large Cap)": "SPY",
    "NASDAQ-100 (US Tech Growth)": "QQQ",
    "Dow Jones 30 (Blue Chip)": "DIA",
    "Dividend Aristocrats (Stable Yield)": "NOBL",

    # 🌍 Global Equity
    "MSCI World (Global Leaders)": "URTH",
    "STOXX 50 (Europe Mega Cap)": "FEZ",

    # 🌏 Asia Markets
    "Nikkei 225 (Japan Blue Chip)": "EWJ",
    "NIFTY 50 (India Large Cap)": "INDY",

    # 🛢️ Sector ETFs (better screening by sector)
    "US Technology": "XLK",
    "US Financials": "XLF",
    "US Healthcare": "XLV",
    "US Industrials": "XLI",
    "US Energy": "XLE",
    "US Consumer Discretionary": "XLY",
    "US Consumer Staples": "XLP",
    "US Utilities": "XLU",
    "US Real Estate": "XLRE",

    # 🪙 Crypto
    "Crypto - Top Market Cap": "BTC-USD,ETH-USD,SOL-USD,BNB-USD,AVAX-USD,XRP-USD"

}


def get_dynamic_watchlist(index_name: str):
    """Returns list of tickers for selected index."""
    index_symbol = INDEX_MAP.get(index_name, "^NDX")
    ticker_obj = yf.Ticker(index_symbol)

    _ = ticker_obj.history(period="1d")  # force metadata fetching
    components = ticker_obj.info.get("components")

    if not components:
        return []

    return [c.upper() for c in components]


def compute_one_year_return(ticker: str):
    df = get_stock_data(ticker, "1y")

    if df.empty or len(df) < 100:
        return None

    start = df["Close"].iloc[0]
    end = df["Close"].iloc[-1]

    ret_pct = ((end - start) / start) * 100
    return round(ret_pct, 2), round(end, 2)


def get_top_performers_1y(index_name: str, top_n: int = 10):
    tickers = get_dynamic_watchlist(index_name)
    if not tickers:
        return pd.DataFrame()

    results = []
    for t in tickers:
        val = compute_one_year_return(t)
        if val:
            ret, price = val
            results.append({"Ticker": t, "Return %": ret, "Price": price})

    df = pd.DataFrame(results)
    return df.sort_values("Return %", ascending=False).head(top_n)
