"""ALX-4: parses an inbound TradingView alert webhook payload. TradingView
lets a user template their own JSON body per alert (no fixed schema,
unlike Plaid/Stripe's own webhook formats) -- this accepts the common
conventions (a top-level "ticker"/"symbol" field, an optional "price"/
"close", an optional "action"/"side", and a free-text "message"/
"comment") rather than requiring one exact shape.

Pure/dependency-free (no DB, no network) so it's unit-testable against
a raw payload the same way web/backend/plaid_webhook_verify.py's JWT
verification is -- the DB-aware token lookup and trades-row insert live
in web/backend/routers/tradingview_webhook.py.
"""

from __future__ import annotations

import json
from typing import Optional

_TICKER_KEYS = ("ticker", "symbol", "Ticker", "Symbol")
_PRICE_KEYS = ("price", "close", "Price", "Close")
_MESSAGE_KEYS = ("message", "comment", "alert", "text")
_ACTION_KEYS = ("action", "side", "signal")
_BUY_WORDS = ("buy", "long")
_SELL_WORDS = ("sell", "short")


class TradingViewAlertError(Exception):
    """Raised when the payload has no identifiable ticker -- a journal
    entry needs one, and this app never guesses a ticker from free text."""


def _infer_direction(payload: dict, message: str) -> Optional[str]:
    for key in _ACTION_KEYS:
        value = str(payload.get(key) or "").lower()
        if any(w in value for w in _BUY_WORDS):
            return "LONG"
        if any(w in value for w in _SELL_WORDS):
            return "SHORT"
    lowered = message.lower()
    if any(w in lowered for w in _BUY_WORDS):
        return "LONG"
    if any(w in lowered for w in _SELL_WORDS):
        return "SHORT"
    return None


def parse_tradingview_alert(raw_body: bytes) -> dict:
    """Returns {"ticker", "price" (float or None), "direction" ("LONG"/
    "SHORT"/None), "message"}. Raises TradingViewAlertError when no
    ticker can be identified -- the caller must reject the webhook
    before any DB write, same "verify/validate before acting" posture
    web/backend/plaid_webhook_verify.py's docstring describes."""
    text = raw_body.decode("utf-8", errors="replace").strip()
    payload: dict = {}
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                payload = parsed
        except (json.JSONDecodeError, ValueError):
            payload = {}

    ticker = None
    for key in _TICKER_KEYS:
        if payload.get(key):
            ticker = str(payload[key]).strip().upper()
            break
    if not ticker:
        raise TradingViewAlertError("No ticker/symbol field found in the alert payload.")

    price = None
    for key in _PRICE_KEYS:
        if payload.get(key) is not None:
            try:
                price = float(payload[key])
            except (TypeError, ValueError):
                price = None
            break

    message = None
    for key in _MESSAGE_KEYS:
        if payload.get(key):
            message = str(payload[key])
            break
    if message is None:
        message = text if not payload else json.dumps(payload)

    direction = _infer_direction(payload, message)

    return {"ticker": ticker, "price": price, "direction": direction, "message": message}
