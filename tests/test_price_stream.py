"""web/backend/routers/price_stream.py's WebSocket price push (PLT-3):
cookie/JWT/revocation auth and the "only send a changed price" loop.
Exercised against a fake WebSocket object (duck-typed: .cookies, .accept,
.close, .receive_json, .send_json) rather than a real ASGI/TestClient
socket -- same fast, no-network, no-DB style as every other router test in
this suite.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import WebSocketDisconnect

import web.backend.routers.price_stream as price_stream_mod
from web.backend.auth import SESSION_COOKIE_NAME, SESSION_SECRET
from web.backend.routers.price_stream import _authenticate, stream_prices


def _make_token(user_id="user-1", **overrides):
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "email": "user@example.com", "iat": now, "exp": now + timedelta(days=7)}
    payload.update(overrides)
    return jwt.encode(payload, SESSION_SECRET, algorithm="HS256")


async def _not_revoked(*a, **kw):
    return False


async def _revoked(*a, **kw):
    return True


class _FakeWebSocket:
    def __init__(self, cookies=None, incoming=None, headers=None):
        self.cookies = cookies or {}
        self.headers = headers or {}
        self._incoming = list(incoming or [])
        self.sent: list[dict] = []
        self.accepted = False
        self.closed_code = None

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000):
        self.closed_code = code

    async def receive_json(self):
        if not self._incoming:
            raise WebSocketDisconnect()
        return self._incoming.pop(0)

    async def send_json(self, data):
        self.sent.append(data)


# ---------------------------------------------------------------- _authenticate

def test_authenticate_accepts_a_valid_unrevoked_token(monkeypatch):
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: _make_token()})
    user = asyncio.run(_authenticate(ws))
    assert user == {"id": "user-1", "email": "user@example.com"}


def test_authenticate_rejects_a_missing_cookie():
    ws = _FakeWebSocket(cookies={})
    assert asyncio.run(_authenticate(ws)) is None


def test_authenticate_rejects_a_malformed_token():
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: "not-a-jwt"})
    assert asyncio.run(_authenticate(ws)) is None


def test_authenticate_rejects_a_revoked_session(monkeypatch):
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _revoked)
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: _make_token()})
    assert asyncio.run(_authenticate(ws)) is None


def test_authenticate_rejects_a_disallowed_origin(monkeypatch):
    # A cross-site page can still get a browser to attach the session
    # cookie to a WebSocket handshake (CORS doesn't govern WebSockets the
    # way it does fetch/XHR) -- Origin is the one thing this endpoint has
    # to check for itself.
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: _make_token()}, headers={"origin": "https://evil.example"})
    assert asyncio.run(_authenticate(ws)) is None


def test_authenticate_allows_an_origin_on_the_allow_list(monkeypatch):
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)
    allowed = next(iter(price_stream_mod._ALLOWED_ORIGINS))
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: _make_token()}, headers={"origin": allowed})
    assert asyncio.run(_authenticate(ws)) == {"id": "user-1", "email": "user@example.com"}


def test_authenticate_allows_a_missing_origin_header(monkeypatch):
    # A non-browser client (script, native app) sends no Origin header at
    # all -- same fallback posture as _extract_token allowing a bearer
    # token for curl/scripts.
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)
    ws = _FakeWebSocket(cookies={SESSION_COOKIE_NAME: _make_token()})
    assert asyncio.run(_authenticate(ws)) == {"id": "user-1", "email": "user@example.com"}


# ---------------------------------------------------------------- stream_prices

def test_stream_prices_closes_without_accepting_when_unauthenticated():
    ws = _FakeWebSocket(cookies={})
    asyncio.run(stream_prices(ws))
    assert ws.accepted is False
    assert ws.closed_code == 4401


def test_stream_prices_sends_only_changed_prices(monkeypatch):
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)

    prices = iter([100.0, 100.0, 101.0, 101.0])

    async def fake_run_in_threadpool(fn, ticker):
        return next(prices)

    monkeypatch.setattr(price_stream_mod, "run_in_threadpool", fake_run_in_threadpool)

    ws = _FakeWebSocket(
        cookies={SESSION_COOKIE_NAME: _make_token()},
        # subscribe, then three polling ticks that don't themselves change
        # the subscription (not dicts with a "tickers" list)
        incoming=[{"tickers": ["aapl"]}, "tick", "tick", "tick"],
    )
    asyncio.run(stream_prices(ws))

    assert ws.accepted is True
    assert [m["ticker"] for m in ws.sent] == ["AAPL", "AAPL"]
    assert [m["price"] for m in ws.sent] == [100.0, 101.0]  # the repeated 100.0 and 101.0 ticks are never resent


def test_stream_prices_ignores_a_ticker_once_unsubscribed(monkeypatch):
    monkeypatch.setattr(price_stream_mod, "_is_revoked", _not_revoked)

    async def fake_run_in_threadpool(fn, ticker):
        return 50.0

    monkeypatch.setattr(price_stream_mod, "run_in_threadpool", fake_run_in_threadpool)

    ws = _FakeWebSocket(
        cookies={SESSION_COOKIE_NAME: _make_token()},
        incoming=[{"tickers": ["AAPL"]}, {"tickers": []}, "tick"],
    )
    asyncio.run(stream_prices(ws))

    # AAPL's price (50.0) is sent once on the first tick; the second message
    # clears the subscription, so the third tick polls nothing further.
    assert len(ws.sent) == 1
    assert ws.sent[0]["ticker"] == "AAPL"
