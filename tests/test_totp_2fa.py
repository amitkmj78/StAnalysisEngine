"""services/totp_service.py's pure TOTP wrapper, and web/backend/routers/
auth.py's 2FA login short-circuit / enroll / confirm / disable (PLT-6).
Same fake-connection pattern as tests/test_auth_forced_password_reset.py.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import services.totp_service as totp_mod
import web.backend.routers.auth as auth_mod
from web.backend.routers.auth import (
    LoginRequest,
    TwoFactorConfirmRequest,
    TwoFactorDisableRequest,
    TwoFactorVerifyLoginRequest,
    confirm_2fa,
    disable_2fa,
    login,
    verify_login_2fa,
)


def _make_request() -> Request:
    scope = {
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    return Request(scope)


class _FakeConn:
    def __init__(self, fetchrow=None):
        self._fetchrow = fetchrow or (lambda sql, *a: None)

    async def fetchrow(self, sql, *args):
        return self._fetchrow(sql, *args)

    async def execute(self, sql, *args):
        return "OK"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


async def _noop_write_audit(*a, **kw):
    pass


# ---------------------------------------------------------------- totp_service

def test_generate_secret_and_verify_round_trip():
    import pyotp
    secret = totp_mod.generate_secret()
    code = pyotp.TOTP(secret).now()
    assert totp_mod.verify_code(secret, code) is True


def test_verify_code_rejects_a_wrong_code():
    secret = totp_mod.generate_secret()
    assert totp_mod.verify_code(secret, "000000") is False


def test_verify_code_rejects_non_numeric_input():
    secret = totp_mod.generate_secret()
    assert totp_mod.verify_code(secret, "abcdef") is False
    assert totp_mod.verify_code(secret, "") is False


def test_provisioning_uri_contains_issuer_and_secret():
    secret = totp_mod.generate_secret()
    uri = totp_mod.provisioning_uri(secret, "user@example.com")
    assert "StAnalysisEngine" in uri
    assert secret in uri


def test_qr_code_data_uri_is_a_png_data_uri():
    secret = totp_mod.generate_secret()
    uri = totp_mod.provisioning_uri(secret, "user@example.com")
    data_uri = totp_mod.qr_code_data_uri(uri)
    assert data_uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------- login 2FA short-circuit

_PASSWORD_HASH = bcrypt.hashpw(b"correct-password", bcrypt.gensalt()).decode()


def _user_row(**overrides):
    row = {
        "id": "user-1", "email": "user@example.com", "password_hash": _PASSWORD_HASH,
        "approved": True, "is_active": True, "role": "user", "must_reset_password": False,
        "totp_enabled": False, "totp_secret_encrypted": None,
    }
    row.update(overrides)
    return row


def test_login_withholds_session_when_2fa_is_enabled(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: _user_row(totp_enabled=True))
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    result = asyncio.run(login(_make_request(), LoginRequest(email="user@example.com", password="correct-password"), Response()))
    assert result["requires_2fa"] is True
    assert "partial_token" in result
    assert "token" not in result

    payload = jwt.decode(result["partial_token"], auth_mod.SESSION_SECRET, algorithms=["HS256"])
    assert payload["purpose"] == "2fa_pending"
    assert payload["sub"] == "user-1"


def test_login_issues_a_normal_session_when_2fa_is_disabled(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: _user_row(totp_enabled=False))
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    result = asyncio.run(login(_make_request(), LoginRequest(email="user@example.com", password="correct-password"), Response()))
    assert "token" in result
    assert "requires_2fa" not in result


# ---------------------------------------------------------------- verify_login_2fa

def test_verify_login_2fa_succeeds_with_the_right_code(monkeypatch):
    secret = totp_mod.generate_secret()
    encrypted = b"fake-encrypted-blob"
    conn = _FakeConn(fetchrow=lambda sql, *a: {
        "id": "user-1", "email": "user@example.com", "role": "user",
        "totp_enabled": True, "totp_secret_encrypted": encrypted,
    })
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "decrypt_token", lambda blob: secret)
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    import pyotp
    code = pyotp.TOTP(secret).now()
    now = datetime.now(timezone.utc)
    partial_token = jwt.encode(
        {"sub": "user-1", "purpose": "2fa_pending", "iat": now, "exp": now + timedelta(minutes=5)},
        auth_mod.SESSION_SECRET, algorithm="HS256",
    )
    result = asyncio.run(verify_login_2fa(
        _make_request(), TwoFactorVerifyLoginRequest(partial_token=partial_token, code=code), Response()
    ))
    assert "token" in result
    assert result["email"] == "user@example.com"


def test_verify_login_2fa_rejects_a_wrong_code(monkeypatch):
    secret = totp_mod.generate_secret()
    conn = _FakeConn(fetchrow=lambda sql, *a: {
        "id": "user-1", "email": "user@example.com", "role": "user",
        "totp_enabled": True, "totp_secret_encrypted": b"blob",
    })
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "decrypt_token", lambda blob: secret)
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    now = datetime.now(timezone.utc)
    partial_token = jwt.encode(
        {"sub": "user-1", "purpose": "2fa_pending", "iat": now, "exp": now + timedelta(minutes=5)},
        auth_mod.SESSION_SECRET, algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_login_2fa(
            _make_request(), TwoFactorVerifyLoginRequest(partial_token=partial_token, code="000000"), Response()
        ))
    assert exc.value.status_code == 401


def test_verify_login_2fa_rejects_an_expired_token():
    now = datetime.now(timezone.utc)
    expired_token = jwt.encode(
        {"sub": "user-1", "purpose": "2fa_pending", "iat": now - timedelta(minutes=10), "exp": now - timedelta(minutes=5)},
        auth_mod.SESSION_SECRET, algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_login_2fa(_make_request(), TwoFactorVerifyLoginRequest(partial_token=expired_token, code="123456"), Response()))
    assert exc.value.status_code == 401


def test_verify_login_2fa_rejects_a_token_with_the_wrong_purpose():
    now = datetime.now(timezone.utc)
    wrong_purpose_token = jwt.encode(
        {"sub": "user-1", "purpose": "something_else", "iat": now, "exp": now + timedelta(minutes=5)},
        auth_mod.SESSION_SECRET, algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(verify_login_2fa(_make_request(), TwoFactorVerifyLoginRequest(partial_token=wrong_purpose_token, code="123456"), Response()))
    assert exc.value.status_code == 401


# ---------------------------------------------------------------- confirm_2fa / disable_2fa

def test_confirm_2fa_enables_only_on_a_correct_code(monkeypatch):
    secret = totp_mod.generate_secret()
    conn = _FakeConn(fetchrow=lambda sql, *a: {"totp_secret_encrypted": b"blob"})
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "decrypt_token", lambda blob: secret)
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    async def fake_verify_bearer_token(request):
        return {"id": "user-1", "email": "user@example.com"}
    monkeypatch.setattr(auth_mod, "verify_bearer_token", fake_verify_bearer_token)

    import pyotp
    code = pyotp.TOTP(secret).now()
    result = asyncio.run(confirm_2fa(_make_request(), TwoFactorConfirmRequest(code=code)))
    assert result == {"totp_enabled": True}


def test_confirm_2fa_rejects_a_wrong_code(monkeypatch):
    secret = totp_mod.generate_secret()
    conn = _FakeConn(fetchrow=lambda sql, *a: {"totp_secret_encrypted": b"blob"})
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "decrypt_token", lambda blob: secret)

    async def fake_verify_bearer_token(request):
        return {"id": "user-1", "email": "user@example.com"}
    monkeypatch.setattr(auth_mod, "verify_bearer_token", fake_verify_bearer_token)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(confirm_2fa(_make_request(), TwoFactorConfirmRequest(code="000000")))
    assert exc.value.status_code == 422


def test_disable_2fa_requires_the_correct_password(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: {"password_hash": _PASSWORD_HASH})
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(auth_mod, "write_audit", _noop_write_audit)

    async def fake_verify_bearer_token(request):
        return {"id": "user-1", "email": "user@example.com"}
    monkeypatch.setattr(auth_mod, "verify_bearer_token", fake_verify_bearer_token)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(disable_2fa(_make_request(), TwoFactorDisableRequest(password="wrong-password")))
    assert exc.value.status_code == 401

    result = asyncio.run(disable_2fa(_make_request(), TwoFactorDisableRequest(password="correct-password")))
    assert result == {"totp_enabled": False}
