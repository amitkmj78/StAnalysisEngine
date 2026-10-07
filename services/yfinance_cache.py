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


def get_cached_history(
    ticker: str, period: str, auto_adjust: bool | None = None, interval: str | None = None
) -> pd.DataFrame:
    """Price history for every chart, screen and backtest. Follows the admin price-source switch: Yahoo through
    yfinance, or Alpaca's bars. The provider is part of the cache key, so a switch never serves the other source's data."""
    from .price_provider import get_price_provider

    provider = get_price_provider()
    if provider == "alpaca":
        return _alpaca_history(ticker, period, interval)
    return _yahoo_history(ticker, period, auto_adjust, interval)


def get_cached_history_yahoo_only(
    ticker: str, period: str, auto_adjust: bool | None = None, interval: str | None = None
) -> pd.DataFrame:
    """Same as get_cached_history, but always Yahoo, regardless of the
    admin price-source switch -- for index/ratio tickers (^VIX, ^TNX, ...)
    Alpaca has no concept of at all, same reasoning get_cached_info's own
    docstring gives for staying Yahoo-only. Switching the admin's stock
    price source to Alpaca was observed to silently break
    services/market_data_service.py's internals fetch this way (^VIX/
    ^VIX3M always missing), since get_cached_history's dispatch doesn't
    know these aren't ordinary equity tickers."""
    return _yahoo_history(ticker, period, auto_adjust, interval)


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def _alpaca_history(ticker: str, period: str, interval: str | None) -> pd.DataFrame:
    from .alpaca_client import get_alpaca_history

    return get_alpaca_history(ticker, period, interval)


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def _yahoo_history(ticker: str, period: str, auto_adjust: bool | None, interval: str | None) -> pd.DataFrame:
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


# STR-1's historical-replay stress tests need a FIXED date range (e.g.
# 2008-09-01..2009-03-09), not a trailing period string -- a closed
# historical window never changes, so a much longer TTL than
# CACHE_TTL_SECONDS is safe and avoids re-fetching the same range on
# every stress-test run. Finite (not infinite) to keep this module's one
# caching mechanism (@ttl_cache everywhere) rather than introducing a
# second cache type for one feature.
RANGE_CACHE_TTL_SECONDS = 60 * 60 * 24 * 180  # ~6 months


@ttl_cache(maxsize=256, ttl_seconds=RANGE_CACHE_TTL_SECONDS)
def get_cached_history_range(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Shared yf.Ticker(ticker).history(start=start, end=end, ...) -- new
    plumbing for STR-1's historical-replay stress tests. Nothing else in
    this module supports a date-range fetch (get_cached_history above
    only accepts a trailing period string); the only prior start=/end=
    usage anywhere in this repo is services/trade_storage.py's uncached
    legacy sqlite path. start/end are ISO date strings ("2008-09-01").
    Does NOT fail open (same as get_cached_history above) -- callers wrap
    this in their own per-ticker try/except in a bounded fan-out, same
    convention as portfolio_health_service.py::_fetch_close_for_period."""

    def _fetch():
        return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=auto_adjust)

    return fetch_with_backoff(_fetch).dropna()


@ttl_cache(maxsize=512, ttl_seconds=CACHE_TTL_SECONDS)
def get_earnings_report_dates(ticker: str) -> list:
    """STB-1: the dates of this stock's last ~6 years of reported earnings, for the backtest's earnings-window rule.
    Only rows that have an actual reported EPS count (upcoming dates are excluded). Empty list when unavailable."""
    try:
        result = fetch_with_backoff(lambda: yf.Ticker(ticker).get_earnings_dates(limit=24))
    except Exception:
        return []
    if result is None or result.empty or "Reported EPS" not in result.columns:
        return []
    return [pd.Timestamp(ts) for ts in result[result["Reported EPS"].notna()].index]


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


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_fund_top_holdings(ticker: str) -> pd.DataFrame:
    """HLT-1: Shared yf.Ticker(ticker).funds_data.top_holdings -- a fund's
    top 10 disclosed holdings, DataFrame indexed by Symbol with a
    "Holding Percent" column (fraction of 1). Broad except (not just
    yfinance's own YFDataException) since accessing .funds_data on a
    non-fund ticker (a plain stock) raises YFDataException ("No Fund
    data found."), while an invalid ticker raises a plain requests.
    HTTPError 404 at the .funds_data property access itself -- neither
    is worth distinguishing from any other fetch failure here. Empty
    DataFrame (never None) both for "this is a stock" and for "this is
    a fund with nothing disclosed" (e.g. GLD -- physical gold, genuinely
    0 rows, not a failure)."""
    try:
        result = fetch_with_backoff(lambda: yf.Ticker(ticker).funds_data.top_holdings)
        return result if result is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_eps_trend(ticker: str) -> pd.DataFrame:
    """Shared yf.Ticker(ticker).eps_trend -- new plumbing (SCR-1's
    earnings-revisions factor): indexed by period (0q/+1q/0y/+1y), with
    current/7daysAgo/30daysAgo/60daysAgo/90daysAgo EPS-estimate columns.
    Empty DataFrame (never None) when yfinance has nothing, same
    fail-open convention as every other function in this module."""
    try:
        result = fetch_with_backoff(lambda: yf.Ticker(ticker).eps_trend)
        return result if result is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@ttl_cache(maxsize=1024, ttl_seconds=CACHE_TTL_SECONDS)
def get_cached_ticker_news(ticker: str, lookback_days: int = 14) -> list[dict]:
    """Shared yf.Ticker(ticker).news, normalized to [{"title",
    "published_at"}], filtered to the last lookback_days -- the
    per-ticker analog of services/market_news_service.py's broad-market
    feed (same content.get("title")/content.get("pubDate") extraction),
    used by the trading agent's AI reviewer (AGT-21/24) for dated
    headline grounding. Empty list (never raises) on any failure, same
    fail-open convention as every other function in this module."""
    try:
        raw = fetch_with_backoff(lambda: yf.Ticker(ticker).news)
    except Exception:
        return []

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    items: list[dict] = []
    for entry in raw or []:
        content = entry.get("content") or {}
        title = content.get("title")
        if not title:
            continue
        pub = content.get("pubDate")
        if pub:
            try:
                if pd.Timestamp(pub) < cutoff:
                    continue
            except (ValueError, TypeError):
                pass  # unparseable date -- keep the item rather than drop it silently
        items.append({"title": title, "published_at": pub})
    return items
