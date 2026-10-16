"""Admin-created accounts (web/backend/routers/admin_users.py::create_user)
start on a known default password and must_reset_password=true. login()
must refuse to issue a session for such an account and instead hand back a
one-time reset token (the same mechanism forgot-password uses); only
reset_password actually changes the password, and must also clear the flag.
Same fake-connection pattern as tests/test_admin_roles_and_theme.py.
"""

import asyncio

import bcrypt
import pytest
from fastapi import HTTPException, Response
from starlette.requests import Request

import web.backend.routers.auth as auth_mod
from web.backend.routers.auth import LoginRequest, ResetPasswordRequest, login, reset_password


class _FakeConn:
    def __init__(self, fetchrow=None):
        self._fetchrow = fetchrow or (lambda sql, *a: None)
        self.executed: list[tuple] = []

    async def fetchrow(self, sql, *args):
        self.executed.append((sql, args))
        return self._fetchrow(sql, *args)

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _make_request() -> Request:
    scope = {
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    return Request(scope)


_PASSWORD_HASH = bcrypt.hashpw(b"Welcome1!", bcrypt.gensalt()).decode()


def _user_row(**overrides):
    row = {
        "id": "user-1",
        "email": "new@example.com",
        "password_hash": _PASSWORD_HASH,
        "approved": True,
        "is_active": True,
        "role": "user",
        "must_reset_password": False,
        "totp_enabled": False,
        "totp_secret_encrypted": None,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------- login / must_reset_password

def test_login_with_must_reset_password_returns_a_reset_token_without_a_session(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: _user_row(must_reset_password=True))
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))

    async def fake_issue_reset_token(c, user_id):
        assert user_id == "user-1"
        return "one-time-token"

    monkeypatch.setattr(auth_mod, "issue_password_reset_token", fake_issue_reset_token)

    response = Response()
    result = asyncio.run(login(_make_request(), LoginRequest(email="new@example.com", password="Welcome1!"), response))

    assert result == {"must_reset_password": True, "email": "new@example.com", "reset_token": "one-time-token"}
    # No session cookie set, and last_login_at never touched -- the account
    # gets no usable access until it actually resets its password.
    assert "set-cookie" not in {k.decode().lower() for k, v in response.raw_headers}
    assert not any("last_login_at" in sql for sql, _ in conn.executed)


def test_login_without_must_reset_password_issues_a_normal_session(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: _user_row(must_reset_password=False))
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))

    response = Response()
    result = asyncio.run(login(_make_request(), LoginRequest(email="new@example.com", password="Welcome1!"), response))

    assert result["email"] == "new@example.com"
    assert "token" in result
    assert any("last_login_at" in sql for sql, _ in conn.executed)


def test_login_rejects_the_wrong_password_even_for_a_must_reset_account(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: _user_row(must_reset_password=True))
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(login(_make_request(), LoginRequest(email="new@example.com", password="wrong-password"), Response()))
    assert exc.value.status_code == 401


# ---------------------------------------------------------------- reset_password

def test_reset_password_clears_the_must_reset_password_flag(monkeypatch):
    async def no_op_validate(password):
        return None

    monkeypatch.setattr(auth_mod, "_validate_new_password", no_op_validate)

    conn = _FakeConn(fetchrow=lambda sql, *a: {"id": "tok-row", "user_id": "user-1"})
    monkeypatch.setattr(auth_mod, "service_conn", lambda: _FakeConnCtx(conn))

    result = asyncio.run(
        reset_password(_make_request(), ResetPasswordRequest(token="raw-token", password="NewPassw0rd!"))
    )

    assert result == {"message": "Password updated. You can now log in with your new password."}
    update_sql, update_args = next(sql_args for sql_args in conn.executed if "UPDATE users" in sql_args[0])
    assert "must_reset_password = false" in update_sql
    assert update_args[1] == "user-1"
