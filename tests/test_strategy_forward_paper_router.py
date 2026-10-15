"""web/backend/routers/strategy_forward_paper.py: ownership checks,
double-link rejection, and the enable-before-disclosure guard. Same fake-
connection pattern as tests/test_admin_roles_and_theme.py."""

import asyncio

import asyncpg
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import web.backend.routers.strategy_forward_paper as router_mod
from web.backend.routers.strategy_forward_paper import (
    KillSwitchRequest,
    LinkRequest,
    disable_forward_paper,
    enable_forward_paper,
    link_forward_paper,
    set_forward_paper_kill_switch,
    unlink_forward_paper,
)


class _FakeConn:
    def __init__(self, fetchval=None, fetchrow=None):
        self._fetchval = fetchval or (lambda sql, *a: None)
        self._fetchrow = fetchrow or (lambda sql, *a: None)

    async def fetchval(self, sql, *args):
        return self._fetchval(sql, *args)

    async def fetchrow(self, sql, *args):
        return self._fetchrow(sql, *args)


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _request(user_id="u1") -> Request:
    scope = {
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    request = Request(scope)
    request.state.user = {"id": user_id}
    return request


def test_link_rejects_a_strategy_the_caller_does_not_own(monkeypatch):
    # get_account is a SYNC function, called via run_in_threadpool in
    # production -- the fake must be sync too.
    monkeypatch.setattr(router_mod.alpaca_trading_client, "get_account",
                        lambda key_id, secret_key: {"id": "alpaca-acct", "account_number": "123"})

    # saved_strategies lookup (ownership, RLS-scoped) returns nothing.
    conn = _FakeConn(fetchval=lambda sql, *a: None)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(link_forward_paper(_request(), 10, LinkRequest(api_key_id="k", api_secret_key="s")))
    assert exc.value.status_code == 404


def test_link_rejects_a_duplicate_link_with_409(monkeypatch):
    monkeypatch.setattr(router_mod.alpaca_trading_client, "get_account",
                        lambda key_id, secret_key: {"id": "alpaca-acct", "account_number": "123"})
    monkeypatch.setattr(router_mod, "encrypt_token", lambda secret: b"encrypted")

    def fetchrow(sql, *a):
        raise asyncpg.UniqueViolationError("duplicate")

    conn = _FakeConn(fetchval=lambda sql, *a: 10, fetchrow=fetchrow)  # strategy exists and is owned
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(link_forward_paper(_request(), 10, LinkRequest(api_key_id="k", api_secret_key="s")))
    assert exc.value.status_code == 409


def test_enable_requires_disclosure_accepted_first(monkeypatch):
    account_row = {
        "id": 1, "saved_strategy_id": 10, "api_key_id": "k", "alpaca_account_id": None, "account_number": None,
        "status": "active", "enabled": False, "kill_engaged": False, "disclosure_accepted_at": None,
        "last_run_at": None, "last_run_status": None, "created_at": None,
    }
    conn = _FakeConn(fetchrow=lambda sql, *a: account_row)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(enable_forward_paper(_request(), 10))
    assert exc.value.status_code == 403


def test_enable_succeeds_once_disclosure_is_accepted(monkeypatch):
    account_row = {
        "id": 1, "saved_strategy_id": 10, "api_key_id": "k", "alpaca_account_id": None, "account_number": None,
        "status": "active", "enabled": False, "kill_engaged": False, "disclosure_accepted_at": "2026-01-01",
        "last_run_at": None, "last_run_status": None, "created_at": None,
    }
    calls = {"n": 0}

    def fetchrow(sql, *a):
        calls["n"] += 1
        if "UPDATE" in sql:
            return {**account_row, "enabled": True}
        return account_row

    conn = _FakeConn(fetchrow=fetchrow)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    result = asyncio.run(enable_forward_paper(_request(), 10))
    assert result["account"]["enabled"] is True


def test_disable_404s_when_no_account_is_linked(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: None)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(disable_forward_paper(_request(), 10))
    assert exc.value.status_code == 404


def test_kill_switch_updates_the_row(monkeypatch):
    account_row = {
        "id": 1, "saved_strategy_id": 10, "api_key_id": "k", "alpaca_account_id": None, "account_number": None,
        "status": "active", "enabled": True, "kill_engaged": True, "disclosure_accepted_at": "2026-01-01",
        "last_run_at": None, "last_run_status": None, "created_at": None,
    }
    conn = _FakeConn(fetchrow=lambda sql, *a: account_row)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    result = asyncio.run(set_forward_paper_kill_switch(_request(), 10, KillSwitchRequest(engaged=True)))
    assert result["account"]["kill_engaged"] is True


def test_unlink_404s_when_no_account_is_linked(monkeypatch):
    conn = _FakeConn(fetchval=lambda sql, *a: None)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(unlink_forward_paper(_request(), 10))
    assert exc.value.status_code == 404


def test_unlink_succeeds_when_an_account_exists(monkeypatch):
    conn = _FakeConn(fetchval=lambda sql, *a: 1)
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    result = asyncio.run(unlink_forward_paper(_request(), 10))
    assert result == {"ok": True}
