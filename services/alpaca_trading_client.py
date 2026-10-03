"""
Thin wrapper around Alpaca's Trading API, paper environment only
(https://paper-api.alpaca.markets/v2) -- there is no live-trading base URL
constant anywhere in this module, so there is no live/paper toggle to
accidentally flip.

Unlike services/alpaca_client.py (market data, one shared app-wide key
pair from env vars), every function here takes the CALLER's OWN decrypted
Alpaca key pair as an argument -- these are per-user credentials for a
per-user paper account, never read from os.environ.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ALPACA_PAPER_TRADING_BASE_URL = "https://paper-api.alpaca.markets/v2"

_TIMEOUT = 10.0


class AlpacaTradingError(Exception):
    """A definite, non-2xx response from Alpaca's Trading API -- distinct
    from a network/timeout failure (httpx.HTTPError, raised uncaught) so a
    caller can tell "Alpaca rejected this" from "we don't know what
    happened," which is exactly the fork TRD-20/NFR-2's idempotent-retry
    logic needs to make correctly."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Alpaca Trading API error {status_code}: {message}")


def _headers(key_id: str, secret_key: str) -> dict:
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code >= 400:
        message = response.text
        try:
            body = response.json()
            message = body.get("message", message)
        except ValueError:
            pass
        raise AlpacaTradingError(response.status_code, message)


def get_account(key_id: str, secret_key: str) -> dict:
    """GET /v2/account -- buying power, cash, equity, account id/number.
    Used both to validate a freshly-pasted key pair before it's ever
    stored, and to fetch live balances for pre-trade checks."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/account",
        headers=_headers(key_id, secret_key),
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def get_clock(key_id: str, secret_key: str) -> dict:
    """GET /v2/clock -- {is_open, next_open, next_close}. Used for TRD-7
    session state instead of reimplementing a market calendar."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/clock",
        headers=_headers(key_id, secret_key),
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def submit_order(
    key_id: str,
    secret_key: str,
    *,
    client_order_id: str,
    ticker: str,
    side: str,
    qty: float,
    order_type: str,
    time_in_force: str,
    limit_price: Optional[float] = None,
) -> dict:
    """POST /v2/orders. client_order_id is Alpaca's own dedup key -- Alpaca
    itself refuses a second order submitted with a client_order_id it has
    already seen, which is what makes TRD-20/NFR-2's idempotent retry safe
    without any app-side dedup table."""
    payload = {
        "symbol": ticker,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
        "client_order_id": client_order_id,
    }
    if limit_price is not None:
        payload["limit_price"] = str(limit_price)
    response = httpx.post(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders",
        headers=_headers(key_id, secret_key),
        json=payload,
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def get_order_by_client_order_id(key_id: str, secret_key: str, client_order_id: str) -> Optional[dict]:
    """GET /v2/orders:by_client_order_id -- the ambiguous-failure recovery
    path: query this before ever retrying a submit whose network response
    was lost. Returns None (not an error) when Alpaca has no such order,
    which tells the caller the original submit never reached Alpaca."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders:by_client_order_id",
        headers=_headers(key_id, secret_key),
        params={"client_order_id": client_order_id},
        timeout=_TIMEOUT,
    )
    if response.status_code == 404:
        return None
    _raise_for_status(response)
    return response.json()


def list_open_orders(key_id: str, secret_key: str) -> list[dict]:
    """GET /v2/orders?status=open -- used by the status-polling job."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders",
        headers=_headers(key_id, secret_key),
        params={"status": "open"},
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def get_order(key_id: str, secret_key: str, alpaca_order_id: str) -> dict:
    """GET /v2/orders/{id} -- used to fetch the final state of an order
    that just closed (filled/rejected/cancelled) and dropped off the
    open-orders list before its last status change was reconciled."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders/{alpaca_order_id}",
        headers=_headers(key_id, secret_key),
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def list_positions(key_id: str, secret_key: str) -> list[dict]:
    """GET /v2/positions -- used by the positions/cash sync job."""
    response = httpx.get(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/positions",
        headers=_headers(key_id, secret_key),
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def submit_trailing_stop(
    key_id: str,
    secret_key: str,
    *,
    client_order_id: str,
    ticker: str,
    qty: float,
    trail_percent: float,
) -> dict:
    """POST /v2/orders as a broker-side trailing sell stop (AGT-16). GTC so
    the stop outlives the trading day and keeps protecting the position even
    if this app is down."""
    payload = {
        "symbol": ticker,
        "qty": str(qty),
        "side": "sell",
        "type": "trailing_stop",
        "trail_percent": str(trail_percent),
        "time_in_force": "gtc",
        "client_order_id": client_order_id,
    }
    response = httpx.post(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders",
        headers=_headers(key_id, secret_key),
        json=payload,
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def cancel_order(key_id: str, secret_key: str, alpaca_order_id: str) -> None:
    """DELETE /v2/orders/{id}. Alpaca answers 204 on success; a 422 means the
    order already closed and is surfaced to the caller as AlpacaTradingError."""
    response = httpx.delete(
        f"{ALPACA_PAPER_TRADING_BASE_URL}/orders/{alpaca_order_id}",
        headers=_headers(key_id, secret_key),
        timeout=_TIMEOUT,
    )
    _raise_for_status(response)
