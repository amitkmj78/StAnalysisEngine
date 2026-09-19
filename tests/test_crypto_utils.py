import os

import pytest
from cryptography.fernet import Fernet

from web.backend.crypto_utils import decrypt_token, encrypt_token


@pytest.fixture(autouse=True)
def _encryption_key(monkeypatch):
    monkeypatch.setenv("PLAID_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())


def test_round_trip():
    plaintext = "access-sandbox-1234567890"
    ciphertext = encrypt_token(plaintext)
    assert decrypt_token(ciphertext) == plaintext


def test_ciphertext_is_not_the_plaintext():
    plaintext = "access-sandbox-abcdefg"
    ciphertext = encrypt_token(plaintext)
    assert plaintext.encode() not in ciphertext


def test_decrypting_with_a_different_key_raises(monkeypatch):
    ciphertext = encrypt_token("access-sandbox-xyz")
    monkeypatch.setenv("PLAID_TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(ValueError):
        decrypt_token(ciphertext)


def test_missing_key_env_var_raises(monkeypatch):
    monkeypatch.delenv("PLAID_TOKEN_ENCRYPTION_KEY", raising=False)
    with pytest.raises(KeyError):
        encrypt_token("access-sandbox-abc")
