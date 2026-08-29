import logging
from typing import Dict

import yfinance as yf
import pandas as pd

from .cache_utils import ttl_cache
from .rate_limit_utils import fetch_with_backoff
from .yfinance_cache import get_cached_history

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
    """Fetch historical stock data for a given ticker and period. Routes
    through the shared cross-feature cache (services/yfinance_cache.py)
    so this doesn't re-fetch a ticker another feature already pulled the
    same (ticker, period) history for in the last 15 minutes."""
    if not ticker:
        return pd.DataFrame()
    try:
        return get_cached_history(ticker, period)
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
        return get_cached_history(cleaned, period, auto_adjust=True)
    except Exception as e:
        logger.warning("Error fetching adjusted history for %s: %s", ticker, e)
        return pd.DataFrame()


@ttl_cache(maxsize=256, ttl_seconds=8)
def get_latest_price(ticker: str):
    """
    Get the latest live price for the given ticker.

    Uses yfinance's lightweight `fast_info` quote (a single quote lookup,
    not a full history fetch) so this stays cheap enough to poll from the
    UI every several seconds — `get_stock_data` is itself cached for 5
    minutes, so routing through it here would make "live" price polling
    silently return the same stale number for minutes at a time.

    8s, not shorter: the UI polls every 10s (see CurrentPriceBadge), and a
    TTL close to that poll interval means two badges/users requesting the
    same ticker within the window share one real yfinance call instead of
    each firing its own — real yfinance load, not just latency, at stake.
    """
    if not ticker:
        return None
    try:
        # Short/quick retry, not fetch_with_backoff's 8s default — this is
        # polled every 1-2s by the UI, so there's no point making one
        # request block for 8s when the next poll will just try again
        # naturally. One quick retry after ~1s absorbs the common
        # transient "Too Many Requests" blip; anything longer than that,
        # fail fast and fall through to the get_stock_data path below.
        price = fetch_with_backoff(
            lambda: yf.Ticker(ticker).fast_info.get("lastPrice"),
            max_retries=1, base_delay=0.1, retry_delay=1.0,
        )
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
        # Same short-retry rationale as get_latest_price — polled
        # frequently, so fail fast rather than blocking on the 8s default.
        info = fetch_with_backoff(
            lambda: yf.Ticker(ticker).info,
            max_retries=1, base_delay=0.1, retry_delay=1.0,
        )
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
