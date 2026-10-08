"""ALX-3's wiring into services.notification_dispatcher.dispatch_alert.
Quiet-hours/digest, the webhook channel, and email delivery are already
covered by tests/test_notification_dispatcher.py's pure-function tests;
this exercises the new push branch specifically against a fake
connection spanning dispatch_alert's several separate service_conn()
calls.
"""

import asyncio
from datetime import time

import pytest

import services.notification_dispatcher as dispatcher


class _FakeConn:
    def __init__(self, settings_row=None, push_subs=None, email="user@example.com"):
        self.settings_row = settings_row
        self.push_subs = push_subs or []
        self.email = email
        self.executed: list[tuple] = []

    async def fetchrow(self, sql, *args):
        if "FROM user_notification_settings" in sql:
            return self.settings_row
        if "FROM user_alert_preferences" in sql:
            return None  # no override -- falls back to DEFAULT_PREFERENCE
        return None

    async def fetch(self, sql, *args):
        if "FROM push_subscriptions" in sql:
            return self.push_subs
        return []

    async def fetchval(self, sql, *args):
        if "SELECT email" in sql:
            return self.email
        return None

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "DELETE 1" if "DELETE FROM push_subscriptions" in sql else "INSERT 0 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _sub(id_=1, endpoint="https://push.example.com/abc"):
    return {"id": id_, "endpoint": endpoint, "p256dh_key": "p256", "auth_key": "auth"}


def _settings(push_enabled=True, quiet_start=None, quiet_end=None, webhook_enabled=False):
    return {
        "quiet_hours_start": quiet_start, "quiet_hours_end": quiet_end, "digest_enabled": False,
        "webhook_enabled": webhook_enabled, "webhook_url": None, "webhook_secret": None,
        "push_enabled": push_enabled,
    }


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch):
    monkeypatch.setattr(dispatcher, "send_alert_email", lambda *a, **k: True)


def test_push_sent_when_subscribed_and_enabled(monkeypatch):
    conn = _FakeConn(settings_row=_settings(), push_subs=[_sub()])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))

    sent_calls = []

    def fake_send_push(subscription, title, body, url):
        sent_calls.append((subscription, title, body, url))
        return True, False

    monkeypatch.setattr(dispatcher, "send_push", fake_send_push)

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    assert len(sent_calls) == 1
    assert sent_calls[0][0] == {"endpoint": "https://push.example.com/abc", "keys": {"p256dh": "p256", "auth": "auth"}}


def test_push_skipped_during_quiet_hours(monkeypatch):
    conn = _FakeConn(settings_row=_settings(quiet_start=time(0, 0), quiet_end=time(23, 59)), push_subs=[_sub()])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))

    sent_calls = []
    monkeypatch.setattr(dispatcher, "send_push", lambda *a, **k: sent_calls.append(a) or (True, False))

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    assert sent_calls == []


def test_push_skipped_when_disabled(monkeypatch):
    conn = _FakeConn(settings_row=_settings(push_enabled=False), push_subs=[_sub()])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))

    sent_calls = []
    monkeypatch.setattr(dispatcher, "send_push", lambda *a, **k: sent_calls.append(a) or (True, False))

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    assert sent_calls == []


def test_push_defaults_on_when_no_settings_row_exists_yet(monkeypatch):
    conn = _FakeConn(settings_row=None, push_subs=[_sub()])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))

    sent_calls = []
    monkeypatch.setattr(dispatcher, "send_push", lambda *a, **k: sent_calls.append(a) or (True, False))

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    assert len(sent_calls) == 1


def test_expired_subscription_is_deleted(monkeypatch):
    conn = _FakeConn(settings_row=_settings(), push_subs=[_sub(id_=42)])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(dispatcher, "send_push", lambda *a, **k: (False, True))

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    delete_calls = [c for c in conn.executed if "DELETE FROM push_subscriptions" in c[0]]
    assert len(delete_calls) == 1
    assert delete_calls[0][1] == (42,)


def test_no_subscriptions_means_no_push_attempt(monkeypatch):
    conn = _FakeConn(settings_row=_settings(), push_subs=[])
    monkeypatch.setattr(dispatcher, "service_conn", lambda: _FakeConnCtx(conn))

    sent_calls = []
    monkeypatch.setattr(dispatcher, "send_push", lambda *a, **k: sent_calls.append(a) or (True, False))

    asyncio.run(dispatcher.dispatch_alert("u1", "AAPL", "signal_change", "Subject", "Body"))

    assert sent_calls == []
