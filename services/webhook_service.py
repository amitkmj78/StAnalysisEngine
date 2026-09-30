"""
ALR-3: outbound webhook delivery for power users. No precedent to reuse
in this codebase -- the only existing webhook code (services/
stripe_service.py, services/plaid_client.py) verifies INBOUND webhook
signatures, the opposite direction. Follows this app's established
fail-open-and-log posture (see services/email_service.py::_send_email)
rather than raising or retrying: a broken or slow third-party endpoint
must never block the alert pipeline that's trying to deliver a
different, already-successful notification.
"""

import hashlib
import hmac
import json
import logging
import secrets

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT_SECONDS = 5


def generate_webhook_secret() -> str:
    """A per-user signing secret, generated once server-side the first
    time a user enables their webhook (see web/backend/routers/
    alert_preferences.py) -- shown to them exactly once, like an API key."""
    return secrets.token_hex(32)


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def send_webhook(url: str, secret: str, payload: dict) -> bool:
    """POSTs `payload` as JSON with an X-Signature header (HMAC-SHA256
    over the exact request body, hex-encoded) the receiver can verify.
    Never raises -- a timeout, connection failure, or non-2xx response
    all just log a warning and return False, same contract as
    email_service.py's _send_email."""
    try:
        body = json.dumps(payload).encode()
        signature = _sign(secret, body)
        response = httpx.post(
            url,
            content=body,
            headers={"Content-Type": "application/json", "X-Signature": signature},
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Webhook delivery failed for %s: %s", url, e)
        return False
