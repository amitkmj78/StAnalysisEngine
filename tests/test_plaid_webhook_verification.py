import base64
import hashlib
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from web.backend.plaid_webhook_verify import WebhookVerificationError, verify_plaid_webhook

KEY_ID = "test-key-1"
BODY = b'{"webhook_type": "HOLDINGS", "webhook_code": "DEFAULT_UPDATE", "item_id": "abc123"}'


def _b64url_uint(value: int, length: int = 32) -> str:
    return base64.urlsafe_b64encode(value.to_bytes(length, "big")).rstrip(b"=").decode()


@pytest.fixture()
def keypair():
    """A locally generated EC P-256 keypair standing in for one of
    Plaid's real webhook-signing keys -- private key signs (plays
    Plaid's role), public JWK is what get_verification_key would return
    (plays our cached-lookup role). Same ES256/JWK shape Plaid's real
    /webhook_verification_key/get response uses."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url_uint(numbers.x),
        "y": _b64url_uint(numbers.y),
        "kid": KEY_ID,
        "alg": "ES256",
        "use": "sig",
        "expired_at": None,
    }
    return private_pem, jwk


def _second_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    jwk = {
        "kty": "EC", "crv": "P-256",
        "x": _b64url_uint(numbers.x), "y": _b64url_uint(numbers.y),
        "kid": "other-key", "alg": "ES256", "use": "sig", "expired_at": None,
    }
    return private_pem, jwk


def _sign(private_pem: bytes, body: bytes = BODY, iat: int | None = None, kid: str = KEY_ID) -> str:
    payload = {
        "iat": time.time() if iat is None else iat,
        "request_body_sha256": hashlib.sha256(body).hexdigest(),
    }
    return jwt.encode(payload, private_pem, algorithm="ES256", headers={"kid": kid})


def test_valid_signature_and_matching_body_passes(keypair):
    private_pem, jwk = keypair
    token = _sign(private_pem)
    verify_plaid_webhook(token, BODY, get_verification_key=lambda kid: jwk)  # no exception


def test_tampered_body_rejected(keypair):
    private_pem, jwk = keypair
    token = _sign(private_pem)  # signed for BODY
    with pytest.raises(WebhookVerificationError, match="hash"):
        verify_plaid_webhook(token, b'{"webhook_type": "TAMPERED"}', get_verification_key=lambda kid: jwk)


def test_missing_header_rejected(keypair):
    _, jwk = keypair
    with pytest.raises(WebhookVerificationError, match="Missing"):
        verify_plaid_webhook(None, BODY, get_verification_key=lambda kid: jwk)


def test_unknown_kid_rejected(keypair):
    private_pem, _ = keypair
    token = _sign(private_pem, kid="some-other-key")
    with pytest.raises(WebhookVerificationError, match="No verification key"):
        verify_plaid_webhook(token, BODY, get_verification_key=lambda kid: None)


def test_expired_key_rejected(keypair):
    private_pem, jwk = keypair
    expired_jwk = {**jwk, "expired_at": 1700000000}
    token = _sign(private_pem)
    with pytest.raises(WebhookVerificationError, match="expired"):
        verify_plaid_webhook(token, BODY, get_verification_key=lambda kid: expired_jwk)


def test_stale_iat_rejected_as_possible_replay(keypair):
    private_pem, jwk = keypair
    token = _sign(private_pem, iat=time.time() - 3600)  # 1 hour old
    with pytest.raises(WebhookVerificationError, match="old"):
        verify_plaid_webhook(token, BODY, get_verification_key=lambda kid: jwk)


def test_wrong_signing_key_rejected(keypair):
    _, jwk = keypair
    other_private_pem, _ = _second_keypair()
    forged_token = _sign(other_private_pem)  # signed by a DIFFERENT key than jwk represents
    with pytest.raises(WebhookVerificationError, match="Signature verification failed"):
        verify_plaid_webhook(forged_token, BODY, get_verification_key=lambda kid: jwk)
