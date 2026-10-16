"""Admin-created users with roles (role column + ADMIN_EMAIL kept as a
permanent fallback), and the theme-color preference's palette validation.
Same fake-connection pattern as tests/test_community_ideas_eval.py.
"""

import asyncio

import pytest
from fastapi import HTTPException

import web.backend.admin as admin_mod
import web.backend.routers.admin_settings as admin_settings_mod
import web.backend.routers.admin_users as admin_users_mod
from web.backend.routers.admin_settings import ThemeAccentColorUpdate, set_default_theme_accent_color
from web.backend.routers.admin_users import CreateUserRequest, RoleUpdateRequest, create_user, update_user_role


class _FakeConn:
    def __init__(self, **responses):
        self._responses = responses
        self.executed: list[tuple] = []

    async def fetchval(self, sql, *args):
        return self._responses.get("fetchval")

    async def fetchrow(self, sql, *args):
        self.executed.append((sql, args))
        rows = self._responses.get("fetchrow")
        if callable(rows):
            return rows(sql, args)
        return rows


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _make_request(admin_email="admin@example.com"):
    from starlette.requests import Request
    scope = {
        "type": "http", "method": "PUT", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    request = Request(scope)
    request.state.user = {"id": "admin-id", "email": admin_email}
    return request


async def _noop_write_audit(*a, **kw):
    pass


# ---------------------------------------------------------------- require_admin

def test_require_admin_passes_for_the_hardcoded_admin_email_without_a_db_query(monkeypatch):
    def boom():
        raise AssertionError("should not query the DB for ADMIN_EMAIL")
    monkeypatch.setattr(admin_mod, "service_conn", boom)
    user = asyncio.run(admin_mod.require_admin(user={"id": "u1", "email": admin_mod.ADMIN_EMAIL}))
    assert user["email"] == admin_mod.ADMIN_EMAIL


def test_require_admin_passes_for_a_real_admin_role(monkeypatch):
    conn = _FakeConn(fetchval="admin")
    monkeypatch.setattr(admin_mod, "service_conn", lambda: _FakeConnCtx(conn))
    user = asyncio.run(admin_mod.require_admin(user={"id": "u2", "email": "someone@example.com"}))
    assert user["email"] == "someone@example.com"


def test_require_admin_rejects_a_normal_user_role(monkeypatch):
    conn = _FakeConn(fetchval="user")
    monkeypatch.setattr(admin_mod, "service_conn", lambda: _FakeConnCtx(conn))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(admin_mod.require_admin(user={"id": "u3", "email": "someone@example.com"}))
    assert exc.value.status_code == 403


# ---------------------------------------------------------------- update_user_role

def test_update_user_role_refuses_to_demote_the_admin_email_account(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, args: {"email": admin_mod.ADMIN_EMAIL})
    monkeypatch.setattr(admin_users_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(admin_users_mod, "write_audit", _noop_write_audit)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(update_user_role(_make_request(), "admin-id", RoleUpdateRequest(role="user")))
    assert exc.value.status_code == 400


def test_update_user_role_allows_promoting_a_normal_account(monkeypatch):
    lookups = {"email": "regular@example.com"}
    calls = []

    def fetchrow(sql, args):
        calls.append(sql)
        if "UPDATE" in sql:
            return {"id": "u4", "email": lookups["email"], "role": "admin"}
        return lookups

    conn = _FakeConn(fetchrow=fetchrow)
    monkeypatch.setattr(admin_users_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(admin_users_mod, "write_audit", _noop_write_audit)
    result = asyncio.run(update_user_role(_make_request(), "u4", RoleUpdateRequest(role="admin")))
    assert result == {"id": "u4", "email": "regular@example.com", "role": "admin"}
    assert any("session_invalidated_at" in sql for sql in calls)  # felt on an already-open session, not just next login


def test_update_user_role_rejects_an_invalid_role_value():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(update_user_role(_make_request(), "u5", RoleUpdateRequest(role="superuser")))
    assert exc.value.status_code == 422


# ---------------------------------------------------------------- create_user

def test_create_user_rejects_an_invalid_role_value():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(create_user(CreateUserRequest(email="a@example.com", role="root")))
    assert exc.value.status_code == 422


def test_create_user_is_approved_immediately_and_starts_with_the_requested_role(monkeypatch):
    async def fake_run_in_threadpool(fn, *a, **kw):
        return True

    async def fake_issue_reset_token(conn, user_id):
        assert user_id == "new-id"
        return "reset-token-abc"

    conn = _FakeConn(fetchrow=lambda sql, args: {"id": "new-id", "email": "new@example.com", "role": "admin"})
    monkeypatch.setattr(admin_users_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(admin_users_mod, "issue_password_reset_token", fake_issue_reset_token)
    monkeypatch.setattr(admin_users_mod, "run_in_threadpool", fake_run_in_threadpool)

    result = asyncio.run(create_user(CreateUserRequest(email="NEW@example.com", role="admin")))
    assert result == {
        "id": "new-id",
        "email": "new@example.com",
        "role": "admin",
        "temporary_password": admin_users_mod.DEFAULT_TEMP_PASSWORD,
        "welcome_email_sent": True,
    }
    # Admin-created accounts skip the pending-approval queue -- approved is
    # hardcoded true in the insert itself, not merely defaulted by the column.
    # must_reset_password is hardcoded true too -- the fixed default password
    # is only ever usable to reach the forced reset screen.
    insert_sql, insert_args = conn.executed[0]
    assert "INSERT INTO users" in insert_sql
    assert "true" in insert_sql
    assert "must_reset_password" in insert_sql
    assert insert_args[0] == "new@example.com"  # email lower-cased
    assert insert_args[2] == "admin"  # role passed through
    import bcrypt as _bcrypt
    assert _bcrypt.checkpw(admin_users_mod.DEFAULT_TEMP_PASSWORD.encode(), insert_args[1].encode())


# ---------------------------------------------------------------- theme accent color palette

def test_set_default_theme_accent_color_rejects_a_color_outside_the_curated_palette():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(set_default_theme_accent_color(ThemeAccentColorUpdate(default_theme_accent_color="#123456")))
    assert exc.value.status_code == 422


def test_set_default_theme_accent_color_accepts_a_palette_color(monkeypatch):
    saved = {}

    async def fake_set_setting_str(key, value):
        saved[key] = value

    monkeypatch.setattr(admin_settings_mod, "set_setting_str", fake_set_setting_str)
    color = next(iter(admin_settings_mod.THEME_ACCENT_PALETTE))
    result = asyncio.run(set_default_theme_accent_color(ThemeAccentColorUpdate(default_theme_accent_color=color)))
    assert result == {"default_theme_accent_color": color}
    assert saved[admin_settings_mod.DEFAULT_THEME_ACCENT_KEY] == color
