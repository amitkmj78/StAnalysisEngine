"""
Verifies a Plaid webhook's authenticity before anything acts on it.

Plaid signs every webhook delivery with an ES256 JWT in the
Plaid-Verification header. An unauthenticated POST to the webhook
endpoint must never trigger a sync or any DB write -- this is the
gate that decides "trust this payload" vs. "reject it," built now
(Phase 1) even though the actual POST /plaid/webhook route isn't wired
up until Phase 3, per the plan's security section.

Pure and dependency-free except for jwt/cryptography (no network, no
Plaid SDK import) -- get_verification_key is injected so this stays
unit-testable against a self-signed JWT standing in for Plaid's, exactly
like prediction_accuracy_service.py etc. are kept DB/network-free.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable

import jwt
from jwt import PyJWK

# Plaid's own recommendation: reject a webhook whose JWT was issued more
# than this long ago, so a captured-and-replayed request can't be
# re-delivered indefinitely.
MAX_WEBHOOK_AGE_SECONDS = 5 * 60


class WebhookVerificationError(Exception):
    """Any failure to verify a Plaid webhook. Callers must treat this as
    reject-the-request, never as "probably fine, act on it anyway."""


def verify_plaid_webhook(
    verification_header: str | None,
    raw_body: bytes,
    get_verification_key: Callable[[str], dict],
) -> None:
    """
    Raises WebhookVerificationError on any failure; returns None (no
    exception) when the webhook is genuine and unmodified.

    get_verification_key(key_id) -> Plaid's JWK dict for that key_id
    (services/plaid_client.get_webhook_verification_key, itself cached --
    injected here rather than called directly so this function makes no
    network call of its own).
    """
    if not verification_header:
        raise WebhookVerificationError("Missing Plaid-Verification header.")

    try:
        header = jwt.get_unverified_header(verification_header)
    except jwt.InvalidTokenError as e:
        raise WebhookVerificationError(f"Malformed JWT header: {e}") from e

    if header.get("alg") != "ES256":
        raise WebhookVerificationError(f"Unexpected JWT alg: {header.get('alg')!r}")

    key_id = header.get("kid")
    if not key_id:
        raise WebhookVerificationError("JWT header is missing 'kid'.")

    jwk_dict = get_verification_key(key_id)
    if not jwk_dict:
        raise WebhookVerificationError(f"No verification key found for kid={key_id!r}.")
    if jwk_dict.get("expired_at"):
        raise WebhookVerificationError(f"Verification key {key_id!r} has expired.")

    try:
        public_key = PyJWK.from_json(json.dumps(jwk_dict)).key
        payload = jwt.decode(verification_header, key=public_key, algorithms=["ES256"])
    except jwt.InvalidTokenError as e:
        raise WebhookVerificationError(f"Signature verification failed: {e}") from e

    issued_at = payload.get("iat")
    if issued_at is None or time.time() - issued_at > MAX_WEBHOOK_AGE_SECONDS:
        raise WebhookVerificationError("Webhook JWT is missing iat or too old (possible replay).")

    expected_hash = payload.get("request_body_sha256")
    actual_hash = hashlib.sha256(raw_body).hexdigest()
    if not expected_hash or expected_hash != actual_hash:
        raise WebhookVerificationError("Request body hash does not match the signed JWT.")
