"""ALX-1's DB-orchestration layer: web.backend.condition_alerts_eval.
evaluate_due_condition_alerts. The pure engine (parse/build/evaluate) is
covered thoroughly in tests/test_condition_alert_service.py; this
exercises the round trip against a fake connection -- reads due alerts,
fetches regime/price/score history, evaluates, and dispatches/persists
on a real fire.
"""

import asyncio
import json
from datetime import date
from unittest import mock

import pandas as pd
import pytest

import web.backend.condition_alerts_eval as eval_mod


class _FakeConn:
    def __init__(self, alert_rows, regime_rows, score_rows_by_ticker):
        self._alert_rows = alert_rows
        self._regime_rows = regime_rows
        self._score_rows_by_ticker = score_rows_by_ticker
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        if "FROM condition_alerts" in sql:
            return self._alert_rows
        if "FROM market_regime_daily" in sql:
            return self._regime_rows
        if "FROM stock_scores" in sql:
            return self._score_rows_by_ticker.get(args[0], [])
        raise AssertionError(f"Unexpected query: {sql}")

    async def execute(self, sql, *args):
        assert "UPDATE condition_alerts" in sql
        self.executed.append(args)
        return "UPDATE 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _alert_row(id_, ticker, conditions, combinator="AND", user_id="11111111-1111-1111-1111-111111111111"):
    return {
        "id": id_, "user_id": user_id, "ticker": ticker,
        "conditions": json.dumps(conditions), "combinator": combinator,
    }


def _rising_history(n=260, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series([100.0 + i for i in range(n)], index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": pd.Series([1.0] * n, index=idx)})


@pytest.fixture(autouse=True)
def _no_real_dispatch(monkeypatch):
    calls = []

    async def fake_dispatch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(eval_mod, "dispatch_alert", fake_dispatch)
    return calls


def test_evaluate_due_condition_alerts_fires_and_dispatches(monkeypatch, _no_real_dispatch):
    alert = _alert_row(1, "AAPL", [{"field": "price", "op": ">", "value": 150}])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: _rising_history())
    monkeypatch.setattr(eval_mod, "get_earnings_report_dates", lambda t: [])

    triggered = asyncio.run(eval_mod.evaluate_due_condition_alerts())

    assert triggered == 1
    assert len(conn.executed) == 1
    assert conn.executed[0][0] == 1  # alert id
    assert len(_no_real_dispatch) == 1


def test_evaluate_due_condition_alerts_does_not_fire_when_condition_not_met(monkeypatch, _no_real_dispatch):
    alert = _alert_row(1, "AAPL", [{"field": "price", "op": ">", "value": 999999}])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: _rising_history())
    monkeypatch.setattr(eval_mod, "get_earnings_report_dates", lambda t: [])

    triggered = asyncio.run(eval_mod.evaluate_due_condition_alerts())

    assert triggered == 0
    assert conn.executed == []
    assert _no_real_dispatch == []


def test_evaluate_due_condition_alerts_returns_zero_for_no_due_alerts(monkeypatch):
    conn = _FakeConn([], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))

    assert asyncio.run(eval_mod.evaluate_due_condition_alerts()) == 0


def test_evaluate_due_condition_alerts_skips_ticker_with_no_price_history(monkeypatch, _no_real_dispatch):
    alert = _alert_row(1, "ZZZZ", [{"field": "price", "op": ">", "value": 1}])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: pd.DataFrame())
    monkeypatch.setattr(eval_mod, "get_earnings_report_dates", lambda t: [])

    triggered = asyncio.run(eval_mod.evaluate_due_condition_alerts())

    assert triggered == 0
    assert conn.executed == []


def test_evaluate_due_condition_alerts_skips_unparseable_condition_without_crashing(monkeypatch, _no_real_dispatch, caplog):
    alert = _alert_row(1, "AAPL", [{"field": "not_a_real_field", "op": ">", "value": 1}])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(eval_mod, "get_cached_history", lambda t, period, auto_adjust=True: _rising_history())
    monkeypatch.setattr(eval_mod, "get_earnings_report_dates", lambda t: [])

    triggered = asyncio.run(eval_mod.evaluate_due_condition_alerts())

    assert triggered == 0
    assert conn.executed == []


# --- ALX-2: the intraday job ---


def test_evaluate_due_intraday_condition_alerts_fires_on_a_price_only_condition(monkeypatch, _no_real_dispatch):
    alert = _alert_row(1, "AAPL", [{"field": "price", "op": ">", "value": 150}])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))

    def fake_history(ticker, period, auto_adjust=True, interval=None):
        assert interval == eval_mod.INTRADAY_INTERVAL
        assert period == eval_mod.INTRADAY_PERIOD
        return _rising_history()

    monkeypatch.setattr(eval_mod, "get_cached_history", fake_history)

    triggered = asyncio.run(eval_mod.evaluate_due_intraday_condition_alerts())

    assert triggered == 1
    assert len(conn.executed) == 1
    assert eval_mod.INTRADAY_LATENCY_DISCLOSURE in conn.executed[0][1]
    assert len(_no_real_dispatch) == 1


def test_evaluate_due_intraday_condition_alerts_skips_alerts_with_a_daily_only_field(monkeypatch, _no_real_dispatch):
    # Mixing price with regime makes the whole alert ineligible for the
    # faster job -- it must not even attempt a fetch for this ticker.
    alert = _alert_row(1, "AAPL", [
        {"field": "price", "op": ">", "value": 150},
        {"field": "regime", "op": "is", "value": "Risk-On"},
    ])
    conn = _FakeConn([alert], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(
        eval_mod, "get_cached_history",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fetch for an ineligible alert")),
    )

    triggered = asyncio.run(eval_mod.evaluate_due_intraday_condition_alerts())

    assert triggered == 0
    assert conn.executed == []


def test_evaluate_due_intraday_condition_alerts_returns_zero_for_no_due_alerts(monkeypatch):
    conn = _FakeConn([], [], {})
    monkeypatch.setattr(eval_mod, "service_conn", lambda: _FakeConnCtx(conn))

    assert asyncio.run(eval_mod.evaluate_due_intraday_condition_alerts()) == 0
