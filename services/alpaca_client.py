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

import httpx

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
