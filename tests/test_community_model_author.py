import asyncio
from datetime import date

import web.backend.community_model_author as mod


class _FakeConn:
    def __init__(self, latest_rows, prior_by_ticker, already_published=None):
        self._latest_rows = latest_rows
        self._prior_by_ticker = prior_by_ticker
        self._already_published = already_published or set()
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        assert "FROM stock_scores" in sql
        return self._latest_rows

    async def fetchrow(self, sql, *args):
        ticker = args[0]
        prior_signal = self._prior_by_ticker.get(ticker)
        return {"short_signal": prior_signal} if prior_signal is not None else None

    async def fetchval(self, sql, *args):
        ticker = args[0]
        return 1 if ticker in self._already_published else None

    async def execute(self, sql, *args):
        self.executed.append(args)
        return "INSERT 0 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _row(ticker, short_signal):
    return {"ticker": ticker, "short_signal": short_signal}


def test_publishes_an_idea_for_a_fresh_buy_signal(monkeypatch):
    conn = _FakeConn([_row("AAPL", "Buy")], {"AAPL": "Hold"})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: 150.0)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 1
    assert len(conn.executed) == 1
    args = conn.executed[0]
    assert args[0] == "AAPL"
    assert args[1] == "LONG"


def test_publishes_short_for_a_fresh_trim_signal(monkeypatch):
    conn = _FakeConn([_row("MSFT", "Trim")], {"MSFT": "Buy"})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: 420.0)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 1
    assert conn.executed[0][1] == "SHORT"


def test_no_prior_day_is_not_a_fresh_change(monkeypatch):
    # First-ever captured day for this ticker -- "changed from nothing"
    # would be a spurious idea, same trap signal_change_alert_service.py
    # already solved.
    conn = _FakeConn([_row("NEWCO", "Buy")], {})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: 10.0)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 0
    assert conn.executed == []


def test_unchanged_signal_is_not_republished(monkeypatch):
    conn = _FakeConn([_row("AAPL", "Buy")], {"AAPL": "Buy"})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: 150.0)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 0


def test_already_published_today_is_skipped(monkeypatch):
    conn = _FakeConn([_row("AAPL", "Buy")], {"AAPL": "Hold"}, already_published={"AAPL"})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: 150.0)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 0


def test_no_price_available_is_skipped(monkeypatch):
    conn = _FakeConn([_row("AAPL", "Buy")], {"AAPL": "Hold"})
    monkeypatch.setattr(mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(mod, "get_latest_price", lambda t: None)

    published = asyncio.run(mod.publish_model_ideas_for_today())

    assert published == 0
