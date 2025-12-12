import yfinance as yf
from datetime import datetime


def safe(info, key, default="N/A"):
    """Safely extract fields from Yahoo info dict."""
    try:
        val = info.get(key, default)
        return val if val not in (None, "", "None") else default
    except Exception:
        return default


def get_basic_stock_info(ticker: str) -> str:
    """
    Returns a clean, reliable company snapshot for the given ticker.
    Always returns non-empty markdown suitable for the meta-agent.
    """

    try:
        stock = yf.Ticker(ticker)

        # Primary data source
        info = stock.info or {}

        # If .info is empty (common now), fall back to fast_info
        if not info:
            try:
                info = stock.fast_info or {}
            except Exception:
                info = {}

        if not info:
            return f"⚠️ No data found for ticker **{ticker}**."

        today = datetime.now().strftime("%Y-%m-%d")

        return f"""
## 📌 Basic Company Snapshot — {ticker}  
_As of {today}_  

### 🏢 Company Profile
- **Name:** {safe(info, 'longName')}
- **Sector:** {safe(info, 'sector')}
- **Industry:** {safe(info, 'industry')}

### 💰 Stock Price & Valuation
- **Current Price:** ${safe(info, 'currentPrice')}
- **Market Cap:** {safe(info, 'marketCap')}
- **Trailing P/E:** {safe(info, 'trailingPE')}
- **Forward P/E:** {safe(info, 'forwardPE')}
- **Revenue Per Share:** {safe(info, 'revenuePerShare')}

### 📊 Fundamental Metrics
- **Total Revenue:** {safe(info, 'totalRevenue')}
- **EBITDA:** {safe(info, 'ebitda')}
- **Operating Cashflow:** {safe(info, 'operatingCashflow')}

### 📈 Trading Range
- **52-Week High / Low:** {safe(info, 'fiftyTwoWeekHigh')} / {safe(info, 'fiftyTwoWeekLow')}
- **Day Low / High:** {safe(info, 'regularMarketDayLow')} / {safe(info, 'regularMarketDayHigh')}
- **Previous Close:** {safe(info, 'previousClose')}

### 🎯 Analyst Targets
- **High:** {safe(info, 'targetHighPrice')}
- **Mean:** {safe(info, 'targetMeanPrice')}
- **Low:** {safe(info, 'targetLowPrice')}

---

📝 _Data sourced from Yahoo Finance. Metrics may vary depending on availability._
"""
    except Exception as e:
        return f"❌ Error fetching data for {ticker}: {e}"
