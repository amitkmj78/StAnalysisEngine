import asyncio
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from services.agent import runner
from services.alpaca_trading_client import AlpacaTradingError


def _history_df(days=300, start=50.0, end=80.0):
    close = np.linspace(start, end, days) + np.sin(np.arange(days)) * 0.5
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": np.full(days, 2_000_000.0)})


class FakeBroker:
    def __init__(self, positions=None, open_orders=None, is_open=True, equity=100_000.0,
                 last_equity=100_000.0, cash=100_000.0):
        self.positions = [dict(p) for p in (positions or [])]
        self.open_orders = list(open_orders or [])
        self.is_open = is_open
        self.equity = equity
        self.last_equity = last_equity
        self.cash = cash
        self.calls: list[str] = []
        self.fail_once: dict[str, Exception] = {}
        self._n = 0

    def _next_id(self):
        self._n += 1
        return f"oid-{self._n}"

    async def __call__(self, fn, creds, *args, **kwargs):
        name = fn.__name__
        self.calls.append(name)
        if name in self.fail_once:
            raise self.fail_once.pop(name)
        if name == "get_clock":
            return {"is_open": self.is_open}
        if name == "get_account":
            return {"equity": str(self.equity), "last_equity": str(self.last_equity), "cash": str(self.cash)}
        if name == "list_positions":
            return [dict(p) for p in self.positions]
        if name == "list_open_orders":
            return [dict(o) for o in self.open_orders]
        if name == "submit_order":
            oid = self._next_id()
            if kwargs["side"] == "buy":
                self.positions.append({"symbol": kwargs["ticker"], "qty": str(kwargs["qty"]),
                                       "current_price": "80.0"})
            return {"id": oid}
        if name == "submit_trailing_stop":
            oid = self._next_id()
            self.open_orders.append({"id": oid, "symbol": kwargs["ticker"], "qty": str(kwargs["qty"]),
                                     "side": "sell", "type": "trailing_stop",
                                     "trail_percent": str(kwargs["trail_percent"])})
            return {"id": oid}
        if name == "cancel_order":
            self.open_orders = [o for o in self.open_orders if o["id"] != args[0]]
            return None
        if name == "get_order":
            return {"status": "filled", "filled_qty": "1", "filled_avg_price": "80.0"}
        if name == "get_order_by_client_order_id":
            return None
        raise AssertionError(f"unexpected broker call {name}")


@pytest.fixture
def harness(monkeypatch):
    journal: list[dict] = []
    alerts: list[dict] = []
    flags = {"agent_kill": False, "paper_kill": False}

    async def fake_journal(run_id, user_id, event_type, reason, **fields):
        journal.append({"event_type": event_type, "reason": reason, **fields})

    async def fake_alert(user_id, ticker, alert_type, subject, text_body, values=None):
        alerts.append({"alert_type": alert_type, "subject": subject, "body": text_body})

    async def fake_setting(key, default):
        if key == runner.AGENT_KILL_SWITCH_KEY:
            return flags["agent_kill"]
        if key == runner.PAPER_TRADING_KILL_SWITCH_KEY:
            return flags["paper_kill"]
        return default

    async def noop(*a, **k):
        return None

    async def fake_header(*a, **k):
        return 1

    async def fake_scores():
        return {"AAA": {"ticker": "AAA", "as_of_date": date(2026, 10, 2), "short_score": 80.0,
                        "short_signal": "Buy", "sector_key": "Tech"}}

    async def fake_history(ticker):
        return _history_df()

    async def fake_regime():
        return "Risk-On"

    async def fake_earnings(ticker):
        return False

    async def fake_wait(submitted, creds):
        return None

    async def fake_reconcile(*a, **k):
        return None

    monkeypatch.setattr(runner, "_journal", fake_journal)
    monkeypatch.setattr(runner, "dispatch_alert", fake_alert)
    monkeypatch.setattr(runner, "get_setting_bool", fake_setting)
    monkeypatch.setattr(runner, "_write_run_header", fake_header)
    monkeypatch.setattr(runner, "_save_peak_and_latch", noop)
    monkeypatch.setattr(runner, "_latest_scores", fake_scores)
    monkeypatch.setattr(runner, "_history", fake_history)
    monkeypatch.setattr(runner, "regime_as_of", fake_regime)
    monkeypatch.setattr(runner, "_earnings_blackout", fake_earnings)
    monkeypatch.setattr(runner, "_wait_for_fills", fake_wait)
    monkeypatch.setattr(runner, "_reconcile_submitted", fake_reconcile)
    monkeypatch.setattr(runner, "_creds", lambda account: {"key": "k", "secret": "s"})
    return {"journal": journal, "alerts": alerts, "flags": flags, "monkeypatch": monkeypatch}


def _wire(harness, broker, mode):
    async def settings(user_id):
        return {"user_id": user_id, "enabled": True, "mode": mode, "peak_equity": None, "breaker_latched": False}

    async def account(user_id):
        return {"id": 1, "api_key_id": "k", "api_secret_key_encrypted": b"x"}

    harness["monkeypatch"].setattr(runner, "_load_settings", settings)
    harness["monkeypatch"].setattr(runner, "_load_paper_account", account)
    harness["monkeypatch"].setattr(runner, "_broker", broker)


def _types(harness):
    return [e["event_type"] for e in harness["journal"]]


def test_plan_mode_proposes_but_places_nothing(harness):
    broker = FakeBroker()
    _wire(harness, broker, "plan")
    result = asyncio.run(runner.run_agent_for_user("u1"))
    assert result["status"] == "completed"
    assert "submit_order" not in broker.calls
    assert "submit_trailing_stop" not in broker.calls
    assert "proposed" in _types(harness)
    assert "run_completed" in _types(harness)


def test_live_mode_rejects_every_order_with_the_gate_reason(harness):
    broker = FakeBroker()
    _wire(harness, broker, "live")
    result = asyncio.run(runner.run_agent_for_user("u1"))
    assert result["live_blocked"] is True
    assert "submit_order" not in broker.calls
    rejections = [e for e in harness["journal"] if e["event_type"] == "rejected"]
    assert rejections and all("Live trading is blocked" in e["reason"] for e in rejections)


def test_kill_switch_places_nothing_and_skips(harness):
    harness["flags"]["agent_kill"] = True
    broker = FakeBroker()
    _wire(harness, broker, "paper")
    result = asyncio.run(runner.run_agent_for_user("u1"))
    assert result == {"status": "skipped", "reason": "kill_switch"}
    assert not any(c.startswith(("submit", "cancel")) for c in broker.calls)


def test_closed_market_places_no_orders_in_paper_mode(harness):
    broker = FakeBroker(is_open=False)
    _wire(harness, broker, "paper")
    asyncio.run(runner.run_agent_for_user("u1"))
    assert "submit_order" not in broker.calls


def test_paper_mode_submits_buy_then_protects_it_with_a_trailing_stop(harness):
    broker = FakeBroker()
    _wire(harness, broker, "paper")
    result = asyncio.run(runner.run_agent_for_user("u1"))
    assert result["status"] == "completed"
    assert "submit_order" in broker.calls
    assert "submit_trailing_stop" in broker.calls
    stop_positions = [o for o in broker.open_orders if o["type"] == "trailing_stop"]
    assert stop_positions and stop_positions[0]["side"] == "sell"
    assert "stop_placed" in _types(harness)


def test_failed_run_still_protects_held_positions_and_alerts(harness):
    """AGT-29 fault injection: the first broker position read fails mid-run.
    The held position has no stop, so the failure path must place one and
    send an alert. No buy is attempted."""
    broker = FakeBroker(positions=[{"symbol": "HELD", "qty": "10", "current_price": "70.0"}])
    broker.fail_once["list_positions"] = httpx.ConnectError("broker unreachable")
    _wire(harness, broker, "paper")
    result = asyncio.run(runner.run_agent_for_user("u1"))
    assert result["status"] == "failed"
    stops = [o for o in broker.open_orders if o["type"] == "trailing_stop"]
    assert any(o["symbol"] == "HELD" and o["qty"] == "10" for o in stops)
    assert "submit_order" not in broker.calls
    assert any(a["alert_type"] == "agent_failure" for a in harness["alerts"])
    assert "run_failed" in _types(harness)


def test_kill_switch_keeps_existing_stops_untouched(harness):
    existing = {"id": "stop-1", "symbol": "HELD", "qty": "10", "side": "sell", "type": "trailing_stop"}
    broker = FakeBroker(positions=[{"symbol": "HELD", "qty": "10", "current_price": "70.0"}], open_orders=[existing])
    harness["flags"]["agent_kill"] = True
    _wire(harness, broker, "paper")
    asyncio.run(runner.run_agent_for_user("u1"))
    assert [o["id"] for o in broker.open_orders] == ["stop-1"]
    assert not any(c in ("cancel_order", "submit_trailing_stop", "submit_order") for c in broker.calls)


def test_ambiguous_submit_failure_is_reconciled_not_retried(harness):
    broker = FakeBroker()
    broker.fail_once["submit_order"] = httpx.ReadTimeout("timed out")
    _wire(harness, broker, "paper")
    asyncio.run(runner.run_agent_for_user("u1"))
    assert "submit_order" in broker.calls
    rejected = [e for e in harness["journal"] if e["event_type"] == "rejected"]
    assert any("Network failure before Alpaca accepted" in e["reason"] for e in rejected)
    assert broker.calls.count("submit_order") == 1  # no blind retry


def test_alpaca_rejection_is_journaled_with_reason(harness):
    broker = FakeBroker()
    broker.fail_once["submit_order"] = AlpacaTradingError(422, "insufficient buying power")
    _wire(harness, broker, "paper")
    asyncio.run(runner.run_agent_for_user("u1"))
    assert any("insufficient buying power" in e["reason"] for e in harness["journal"] if e["event_type"] == "rejected")
