from datetime import datetime

from services.data_service import get_latest_price
from services.yfinance_cache import get_cached_info


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

    Current Price comes from get_latest_price (the same live,
    retry/backoff-wrapped, admin-switchable-provider price used
    everywhere else in the app — /predict, /portfolio, /search), not
    yfinance's own `.info['currentPrice']` field. That field is a
    known source of self-contradictory answers: it can be stale
    relative to `.info`'s own regularMarketDayLow/High in the same
    response, producing a "current price" that falls outside the
    day's own range — confusing enough that a user reported it as
    "this information does not make any sense." Everything else here
    (sector, market cap, P/E, targets) still comes from `.info`
    (via the shared cache, not a raw uncached call) since those
    fields don't carry the same same-response self-contradiction risk.
    """

    try:
        info = get_cached_info(ticker)

        if not info:
            return f"⚠️ No data found for ticker **{ticker}**."

        current_price = get_latest_price(ticker)
        current_price_str = f"{current_price:.2f}" if current_price is not None else safe(info, "currentPrice")

        today = datetime.now().strftime("%Y-%m-%d")

        return f"""
## 📌 Basic Company Snapshot — {ticker}
_As of {today}_

### 🏢 Company Profile
- **Name:** {safe(info, 'longName')}
- **Sector:** {safe(info, 'sector')}
- **Industry:** {safe(info, 'industry')}

### 💰 Stock Price & Valuation
- **Current Price:** ${current_price_str}
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

📝 _Current price from this app's live quote feed; other fields from Yahoo Finance. Metrics may vary depending on availability._
"""
    except Exception as e:
        return f"❌ Error fetching data for {ticker}: {e}"
