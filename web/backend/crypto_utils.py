import os

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    # Hard-fail like SESSION_SECRET (auth.py) -- a Plaid access_token is
    # full read access to a real brokerage account, so there is no safe
    # default to fall back to if this isn't configured. Also used to
    # encrypt Alpaca paper-trading API secret keys (web/backend/routers/
    # paper_trading.py) -- same rationale, kept generic rather than a
    # second Fernet key/env var for a second kind of secret.
    return Fernet(os.environ["PLAID_TOKEN_ENCRYPTION_KEY"].encode())


def encrypt_token(plaintext: str) -> bytes:
    return _fernet().encrypt(plaintext.encode())


def decrypt_token(ciphertext: bytes) -> str:
    try:
        return _fernet().decrypt(bytes(ciphertext)).decode()
    except InvalidToken as e:
        raise ValueError("Could not decrypt token -- wrong key or corrupted ciphertext.") from e
