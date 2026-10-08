"""COM-2's DB-orchestration layer: web.backend.community_ideas_eval.
evaluate_due_community_ideas. The pure scoring adapter is covered in
tests/test_community_idea_service.py; this exercises the round trip
against a fake connection and fake price history.
"""

import asyncio
from datetime import date, datetime, timezone

import pandas as pd

import web.backend.community_ideas_eval as eval_mod


class _FakeConn:
    def __init__(self, idea_rows):
        self._idea_rows = idea_rows
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        assert "FROM community_ideas" in sql
        return self._idea_rows

    async def execute(self, sql, *args):
        assert "UPDATE community_ideas" in sql
        self.executed.append(args)
        return "UPDATE 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _idea_row(id_, ticker, direction, created_at, horizon_days=10):
    return {
        "id": id_, "ticker": ticker, "direction": direction,
        "horizon_days": horizon_days, "created_at": created_at,
    }


def _rising_history(n=20, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series([100.0 + i for i in range(n)], index=idx)
    return pd.DataFrame({"Close": close})


def _flat_spy(n=20, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series([500.0] * n, index=idx)


def test_scores_a_matured_long_idea_and_computes_excess_vs_spy(monkeypatch):
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idea = _idea_row(1, "AAPL", "LONG", created_at, horizon_days=10)
    conn = _FakeConn([idea])
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: _rising_history())
    monkeypatch.setattr(eval_mod, "fetch_spy_close_series", lambda: _flat_spy())

    scored = asyncio.run(eval_mod.evaluate_due_community_ideas())

    assert scored == 1
    assert len(conn.executed) == 1
    idea_id, realized, excess, outcome = conn.executed[0]
    assert idea_id == 1
    assert outcome == "hit"  # price rose, LONG
    assert realized > 0
    assert excess == realized  # SPY was flat, so excess == raw realized return


def test_unmatured_idea_is_left_unscored(monkeypatch):
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idea = _idea_row(1, "AAPL", "LONG", created_at, horizon_days=10)
    conn = _FakeConn([idea])
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    # Only 3 bars -- horizon of 10 trading days hasn't elapsed.
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: _rising_history(n=3))
    monkeypatch.setattr(eval_mod, "fetch_spy_close_series", lambda: _flat_spy(n=3))

    scored = asyncio.run(eval_mod.evaluate_due_community_ideas())

    assert scored == 0
    assert conn.executed == []


def test_no_unscored_ideas_returns_zero(monkeypatch):
    conn = _FakeConn([])
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))

    assert asyncio.run(eval_mod.evaluate_due_community_ideas()) == 0


def test_ticker_with_no_price_history_is_skipped(monkeypatch):
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idea = _idea_row(1, "ZZZZ", "LONG", created_at)
    conn = _FakeConn([idea])
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: pd.DataFrame())
    monkeypatch.setattr(eval_mod, "fetch_spy_close_series", lambda: _flat_spy())

    scored = asyncio.run(eval_mod.evaluate_due_community_ideas())

    assert scored == 0
    assert conn.executed == []


def test_short_idea_scored_as_hit_when_price_falls(monkeypatch):
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    idea = _idea_row(1, "AAPL", "SHORT", created_at, horizon_days=10)
    conn = _FakeConn([idea])
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))

    def falling_history():
        idx = pd.bdate_range("2026-01-01", periods=20)
        close = pd.Series([100.0 - i for i in range(20)], index=idx)
        return pd.DataFrame({"Close": close})

    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: falling_history())
    monkeypatch.setattr(eval_mod, "fetch_spy_close_series", lambda: _flat_spy())

    scored = asyncio.run(eval_mod.evaluate_due_community_ideas())

    assert scored == 1
    assert conn.executed[0][3] == "hit"
