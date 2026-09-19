import os

from cryptography.fernet import Fernet, InvalidToken


def _fernet() -> Fernet:
    # Hard-fail like SESSION_SECRET (auth.py) -- a Plaid access_token is
    # full read access to a real brokerage account, so there is no safe
    # default to fall back to if this isn't configured.
    return Fernet(os.environ["PLAID_TOKEN_ENCRYPTION_KEY"].encode())


def encrypt_token(plaintext: str) -> bytes:
    return _fernet().encrypt(plaintext.encode())


def decrypt_token(ciphertext: bytes) -> str:
    try:
        return _fernet().decrypt(bytes(ciphertext)).decode()
    except InvalidToken as e:
        raise ValueError("Could not decrypt token -- wrong key or corrupted ciphertext.") from e
