"""
Thin wrapper around Alpaca's free real-time market data API (IEX feed —
free tier, no funded brokerage account needed, just a free Alpaca account
for API keys). Used as an alternative to yfinance for live quotes; see
services/price_provider.py for how the switch works.

IEX-only means this reflects one exchange's trade tape, not the full
consolidated NBBO — prices can differ from Yahoo's by a cent or two.
Fine for the signals/predictions this app uses live prices for; not
appropriate for order execution.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

ALPACA_DATA_BASE_URL = "https://data.alpaca.markets/v2"


class AlpacaSymbolNotFound(Exception):
    """Alpaca's own 404 "no trade found for {symbol}" — a confirmed,
    permanent answer for instruments that never trade on any exchange
    (mutual funds like FXAIX, CMIUX — Alpaca's IEX feed, like every
    exchange feed, only carries exchange-traded securities). Distinct
    from every other failure mode (network error, auth error, rate
    limit) so a caller can choose to fall back to Yahoo specifically
    here — covering a known, permanent gap, not masking a real Alpaca
    outage the admin-facing switch needs to stay honest about."""


def _headers() -> dict | None:
    key_id = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY")
    if not key_id or not secret_key:
        return None
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}


def get_alpaca_latest_price(ticker: str) -> float | None:
    """Latest real-time trade price for `ticker` on the free IEX feed, or
    None if Alpaca isn't configured or the request fails — fails open
    like every other price lookup in this codebase, so a caller can just
    show "unavailable" rather than crash. Raises AlpacaSymbolNotFound
    (not just None) specifically when Alpaca confirms via a 404 that this
    symbol has no trade data at all — see that class's docstring."""
    headers = _headers()
    if headers is None:
        logger.warning("Alpaca not configured (ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY missing)")
        return None
    try:
        response = httpx.get(
            f"{ALPACA_DATA_BASE_URL}/stocks/{ticker}/trades/latest",
            headers=headers,
            timeout=5.0,
        )
        if response.status_code == 404:
            raise AlpacaSymbolNotFound(ticker)
        response.raise_for_status()
        price = response.json().get("trade", {}).get("p")
        return round(float(price), 2) if price is not None else None
    except AlpacaSymbolNotFound:
        raise
    except Exception as e:
        logger.warning("Error fetching Alpaca latest trade for %s: %s", ticker, e)
        return None


NY_TZ = ZoneInfo("America/New_York")

ALPACA_PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "60d": 60, "1mo": 31, "3mo": 92, "6mo": 182, "1y": 365, "2y": 730, "3y": 1095, "730d": 730, "5y": 1826, "10y": 3652, "max": 7300}  # "max" is capped at 20 years, the reader's limit
ALPACA_TIMEFRAMES = {None: "1Day", "1d": "1Day", "1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
ALPACA_BARS_PAGE_LIMIT = 10000


def _fetch_alpaca_bars(ticker: str, params: dict) -> pd.DataFrame:
    """Shared pagination + frame-shaping for Alpaca's /bars endpoint --
    both get_alpaca_history (trailing period) and get_alpaca_history_range
    (fixed start/end) build their own `params` and call this rather than
    each re-implementing the same page_token loop and column renaming."""
    headers = _headers()
    if headers is None:
        return pd.DataFrame()
    bars: list[dict] = []
    try:
        while True:
            response = httpx.get(f"{ALPACA_DATA_BASE_URL}/stocks/{ticker}/bars", headers=headers, params=params, timeout=15.0)
            if response.status_code == 404:
                return pd.DataFrame()
            response.raise_for_status()
            payload = response.json()
            bars.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
    except Exception as e:
        logger.warning("Alpaca bars fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame(bars)
    frame.index = pd.to_datetime(frame["t"], utc=True).dt.tz_convert("America/New_York")
    frame = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)
    return frame.dropna()


def get_alpaca_history(ticker: str, period: str, interval: str | None = None) -> pd.DataFrame:
    """Price bars from Alpaca's historical API, shaped like the yfinance history frame the rest of the app reads:
    Open, High, Low, Close, Volume, indexed by New York time. Split- and dividend-adjusted for daily bars, like the
    yfinance default. Fails open: an empty frame when Alpaca isn't configured or the request fails, so the caller
    shows "no price history" rather than quietly switching to Yahoo. The IEX feed is used, matching the live quotes."""
    if period not in ALPACA_PERIOD_DAYS or interval not in ALPACA_TIMEFRAMES:
        return pd.DataFrame()
    intraday = interval not in (None, "1d")
    now = datetime.now(timezone.utc)
    start_days = ALPACA_PERIOD_DAYS[period]
    if intraday and period == "1d":
        start_days = 4  # the last session: a few calendar days back covers a weekend or a holiday
    params = {
        "timeframe": ALPACA_TIMEFRAMES[interval],
        "start": (now - timedelta(days=start_days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": ALPACA_BARS_PAGE_LIMIT,
        "adjustment": "all",
        "feed": "iex",
        "sort": "asc",
    }
    frame = _fetch_alpaca_bars(ticker, params)
    if frame.empty:
        return frame
    if intraday and period == "1d":
        last_session = frame.index.normalize().max()
        frame = frame[frame.index.normalize() == last_session]
    return frame


def get_alpaca_history_range(ticker: str, start: str, end: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Price bars for a FIXED historical date range (start/end are ISO
    date strings, e.g. "2008-09-01") rather than a trailing period --
    used by services/yfinance_cache.py::get_cached_history_range for
    STR-1's historical-replay stress tests.

    Real, permanent coverage gap, not a bug: Alpaca's IEX feed only goes
    back to ~2016, so older scenarios (the 2008 GFC, the dot-com bust)
    will always return empty here -- those stress tests stay Yahoo-only
    regardless of this function existing. Scenarios within Alpaca's
    coverage (e.g. the 2020 COVID crash) do benefit."""
    params = {
        "timeframe": "1Day",
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T23:59:59Z",
        "limit": ALPACA_BARS_PAGE_LIMIT,
        "adjustment": "all" if auto_adjust else "raw",
        "feed": "iex",
        "sort": "asc",
    }
    return _fetch_alpaca_bars(ticker, params)


def get_alpaca_previous_close(ticker: str) -> float | None:
    """Yesterday's regular-session close via Alpaca's daily bars -- a
    resilience fallback for services.data_service.get_previous_close
    when Yahoo is unavailable (e.g. rate-limited), not a routine
    alternative source. Reuses get_alpaca_history rather than a second
    bars-fetching implementation.

    Drops today's bar if Alpaca has already started one (requesting
    mid-session would otherwise return today's partial close instead of
    yesterday's real one) -- same reasoning yfinance's own fast_info
    applies to "previousClose" vs "lastPrice". Returns None, never a
    guess, if Alpaca isn't configured or has no data for this ticker
    (e.g. a mutual fund -- Alpaca's exchange feed never carries those,
    same permanent gap AlpacaSymbolNotFound documents for live quotes)."""
    frame = get_alpaca_history(ticker, "5d", interval="1d")
    if frame.empty:
        return None
    today = datetime.now(NY_TZ).date()
    if frame.index[-1].date() == today:
        frame = frame.iloc[:-1]
    if frame.empty:
        return None
    return round(float(frame["Close"].iloc[-1]), 2)
