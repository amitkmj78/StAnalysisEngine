"""ALX-4's DB-aware endpoint (web.backend.routers.tradingview_webhook).
The payload parsing itself is covered in
tests/test_tradingview_webhook_service.py; this exercises the token
lookup (reject before any DB write), the trades-row insert, and the
dispatch_alert call, against a fake connection and a minimal fake
Request (only `.body()` is ever called on it)."""

import asyncio

import pytest
from fastapi import HTTPException

import web.backend.routers.tradingview_webhook as tv_webhook


class _FakeRequest:
    def __init__(self, body: bytes):
        self._body = body

    async def body(self) -> bytes:
        return self._body


class _FakeConn:
    def __init__(self, user_id):
        self._user_id = user_id
        self.executed: list[tuple] = []

    async def fetchval(self, sql, *args):
        assert "tradingview_webhook_token" in sql
        return self._user_id if args[0] == "good-token" else None

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "INSERT 0 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _reset_rate_limit():
    tv_webhook._recent_calls.clear()
    yield
    tv_webhook._recent_calls.clear()


@pytest.fixture(autouse=True)
def _no_real_dispatch(monkeypatch):
    calls = []

    async def fake_dispatch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(tv_webhook, "dispatch_alert", fake_dispatch)
    return calls


def test_valid_token_and_payload_inserts_a_trade_and_dispatches(monkeypatch, _no_real_dispatch):
    conn = _FakeConn("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(tv_webhook, "service_conn", lambda: _FakeConnCtx(conn))
    request = _FakeRequest(b'{"ticker": "AAPL", "price": 150.0, "message": "Buy signal"}')

    result = asyncio.run(tv_webhook.receive_tradingview_alert(request, "good-token"))

    assert result["ok"] is True
    assert result["ticker"] == "AAPL"
    assert len(conn.executed) == 1
    assert "INSERT INTO trades" in conn.executed[0][0]
    assert len(_no_real_dispatch) == 1


def test_unknown_token_rejected_before_any_write(monkeypatch, _no_real_dispatch):
    conn = _FakeConn("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(tv_webhook, "service_conn", lambda: _FakeConnCtx(conn))
    request = _FakeRequest(b'{"ticker": "AAPL"}')

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "bad-token"))

    assert exc_info.value.status_code == 404
    assert conn.executed == []
    assert _no_real_dispatch == []


def test_valid_token_but_unparseable_payload_rejected_as_422(monkeypatch, _no_real_dispatch):
    conn = _FakeConn("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(tv_webhook, "service_conn", lambda: _FakeConnCtx(conn))
    request = _FakeRequest(b'{"message": "no ticker field anywhere"}')

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "good-token"))

    assert exc_info.value.status_code == 422
    assert conn.executed == []


def test_rate_limit_rejects_after_threshold(monkeypatch, _no_real_dispatch):
    conn = _FakeConn("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(tv_webhook, "service_conn", lambda: _FakeConnCtx(conn))
    request = _FakeRequest(b'{"ticker": "AAPL"}')

    for _ in range(tv_webhook._RATE_LIMIT_MAX_PER_WINDOW):
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "good-token"))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "good-token"))
    assert exc_info.value.status_code == 429


def test_rate_limit_is_per_token_not_global(monkeypatch, _no_real_dispatch):
    conn = _FakeConn("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr(tv_webhook, "service_conn", lambda: _FakeConnCtx(conn))
    request = _FakeRequest(b'{"ticker": "AAPL"}')

    for _ in range(tv_webhook._RATE_LIMIT_MAX_PER_WINDOW):
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "good-token"))

    # A different (unknown) token's own budget is untouched by the above --
    # the 404 proves the rate limiter didn't reject it first.
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(tv_webhook.receive_tradingview_alert(request, "another-token"))
    assert exc_info.value.status_code == 404
