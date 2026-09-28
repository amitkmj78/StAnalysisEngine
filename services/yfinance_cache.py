"""
Shared, ticker-keyed cache for yf.Ticker(...).info and .history() — a
single process-wide cache used by every service that fetches these,
instead of each maintaining its own separate fetch (or no cache at all).

Before this, the same ticker's data could be fetched independently by
half a dozen different features within the same minute — e.g. Stock
Finder, the Fund Screener, Goal Plan, and the entry-strategy scanner
each pulling their own copy of AAPL's 1y history — real, avoidable
yfinance load stacked on top of what's already rate-limited. The API
runs as a single uvicorn process (no --workers), so a per-process cache
here fully dedupes across every concurrent request and user, not just
within one.

Deliberately NOT used for live/near-live price lookups (data_service's
get_latest_price / get_extended_hours_price) — those need a much
shorter TTL than this and already have their own.
"""
import pandas as pd
import yfinance as yf

from .cache_utils import ttl_cache
from .rate_limit_utils import fetch_with_backoff

# 15 minutes: long enough to dedupe the same ticker being requested by
# several different features/users within a normal browsing session,
# short enough that nothing relying on "today's" fundamentals/history
# goes meaningfully stale.
CACHE_TTL_SECONDS = 900


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_info(ticker: str) -> dict:
    """Shared yf.Ticker(ticker).info — the heaviest, most-duplicated
    yfinance call in the app."""
    return fetch_with_backoff(lambda: yf.Ticker(ticker).info) or {}


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_history(
    ticker: str, period: str, auto_adjust: bool | None = None, interval: str | None = None
) -> pd.DataFrame:
    """
    Shared yf.Ticker(ticker).history(period=period, ...). `auto_adjust`
    defaults to None (yfinance's own default) rather than True, so
    callers that never specified it keep their exact prior behavior —
    pass True/False explicitly to match what you had before. `interval`
    defaults to None (yfinance's own daily-bar default) too -- pass e.g.
    "5m" for intraday bars (added for DET-1's 1D chart range, the one
    genuinely new piece of this: period="1d" with the default daily
    interval returns a single row, not a chart).

    Keyed on (ticker, period, auto_adjust, interval) — doesn't dedupe
    across different periods for the same ticker (e.g. "1y" vs "3y" are
    cached separately even though "3y" contains "1y"), but that covers
    the common case: most callers already ask for the same handful of
    period values.
    """

    def _fetch():
        kwargs = {}
        if auto_adjust is not None:
            kwargs["auto_adjust"] = auto_adjust
        if interval is not None:
            kwargs["interval"] = interval
        return yf.Ticker(ticker).history(period=period, **kwargs)

    return fetch_with_backoff(_fetch).dropna()


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_earnings_dates(ticker: str) -> pd.DataFrame:
    """Shared yf.Ticker(ticker).get_earnings_dates() -- genuinely new
    plumbing (DET-1's earnings date): confirmed nothing in this app calls
    this or .dividends anywhere today. Index is the earnings datetime;
    upcoming (not-yet-reported) rows have NaN "Reported EPS"/"Surprise(%)"
    -- callers pick the next one relative to now. Empty DataFrame (never
    None) when yfinance has nothing, same fail-open convention as every
    other function in this module."""
    try:
        result = fetch_with_backoff(lambda: yf.Ticker(ticker).get_earnings_dates(limit=8))
        return result if result is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_dividends(ticker: str) -> pd.Series:
    """Shared yf.Ticker(ticker).dividends -- see get_cached_earnings_dates
    above for why this is new plumbing. Empty Series (never None) when
    yfinance has nothing (e.g. a stock that's never paid one)."""
    try:
        result = fetch_with_backoff(lambda: yf.Ticker(ticker).dividends)
        return result if result is not None else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)
