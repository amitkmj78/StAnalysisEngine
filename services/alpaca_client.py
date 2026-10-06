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


ALPACA_PERIOD_DAYS = {"1d": 1, "5d": 5, "7d": 7, "60d": 60, "1mo": 31, "6mo": 182, "1y": 365, "730d": 730, "5y": 1826}
ALPACA_TIMEFRAMES = {None: "1Day", "1d": "1Day", "1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
ALPACA_BARS_PAGE_LIMIT = 10000


def get_alpaca_history(ticker: str, period: str, interval: str | None = None) -> pd.DataFrame:
    """Price bars from Alpaca's historical API, shaped like the yfinance history frame the rest of the app reads:
    Open, High, Low, Close, Volume, indexed by New York time. Split- and dividend-adjusted for daily bars, like the
    yfinance default. Fails open: an empty frame when Alpaca isn't configured or the request fails, so the caller
    shows "no price history" rather than quietly switching to Yahoo. The IEX feed is used, matching the live quotes."""
    headers = _headers()
    if headers is None or period not in ALPACA_PERIOD_DAYS or interval not in ALPACA_TIMEFRAMES:
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
        logger.warning("Alpaca history failed for %s: %s", ticker, e)
        return pd.DataFrame()
    if not bars:
        return pd.DataFrame()
    frame = pd.DataFrame(bars)
    frame.index = pd.to_datetime(frame["t"], utc=True).dt.tz_convert("America/New_York")
    frame = frame.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})[
        ["Open", "High", "Low", "Close", "Volume"]
    ].astype(float)
    if intraday and period == "1d":
        last_session = frame.index.normalize().max()
        frame = frame[frame.index.normalize() == last_session]
    return frame.dropna()
