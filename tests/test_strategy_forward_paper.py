"""services/strategy_forward_paper.py: the daily forward-paper-trading job.
Unit-tests the orchestration/safety logic (kill switches, checks-reuse,
sells-before-buys, ambiguous-failure reconciliation, position state) by
monkeypatching this module's own data-access helpers and the broker client
-- never a real DB connection or a real Alpaca call.
"""

import asyncio

import httpx

import services.strategy_forward_paper as sfp_mod
from services.alpaca_trading_client import AlpacaTradingError


def _account(**overrides):
    base = {
        "id": 1, "saved_strategy_id": 10, "user_id": "u1",
        "api_key_id": "key", "api_secret_key_encrypted": b"enc",
        "enabled": True, "kill_engaged": False, "disclosure_accepted_at": "2026-01-01",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------- run_forward_paper_for_strategy

def test_returns_not_enabled_when_no_account_is_linked(monkeypatch):
    async def fake_load_account(saved_id):
        return None
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)
    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result == {"status": "not_enabled"}


def test_returns_not_enabled_when_the_account_is_disabled(monkeypatch):
    async def fake_load_account(saved_id):
        return _account(enabled=False)
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)
    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result == {"status": "not_enabled"}


def test_returns_not_enabled_when_disclosure_not_accepted(monkeypatch):
    async def fake_load_account(saved_id):
        return _account(disclosure_accepted_at=None)
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)
    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result == {"status": "not_enabled"}


def _patch_journal_helpers(monkeypatch):
    journaled = []

    async def fake_write_run_header(account, status, reason, equity=None):
        journaled.append(("run_header", status, reason))
        return 99

    async def fake_journal(run_id, account, event_type, reason, **fields):
        journaled.append(("journal", event_type, reason))

    async def fake_update_status(account_id, status):
        journaled.append(("status", status))

    monkeypatch.setattr(sfp_mod, "_write_run_header", fake_write_run_header)
    monkeypatch.setattr(sfp_mod, "_journal", fake_journal)
    monkeypatch.setattr(sfp_mod, "_update_account_status", fake_update_status)
    return journaled


def test_row_level_kill_switch_blocks_and_journals_skipped(monkeypatch):
    async def fake_load_account(saved_id):
        return _account(kill_engaged=True)
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)

    async def fake_get_setting_bool(key, default):
        return False
    monkeypatch.setattr(sfp_mod, "get_setting_bool", fake_get_setting_bool)
    journaled = _patch_journal_helpers(monkeypatch)

    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result["status"] == "skipped" and result["reason"] == "kill_switch"
    assert ("status", "skipped") in journaled


def test_global_strategy_kill_switch_blocks(monkeypatch):
    async def fake_load_account(saved_id):
        return _account(kill_engaged=False)
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)

    async def fake_get_setting_bool(key, default):
        return key == sfp_mod.STRATEGY_FORWARD_PAPER_KILL_SWITCH_KEY
    monkeypatch.setattr(sfp_mod, "get_setting_bool", fake_get_setting_bool)
    _patch_journal_helpers(monkeypatch)

    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result["status"] == "skipped" and result["reason"] == "kill_switch"


def test_shared_paper_trading_kill_switch_blocks(monkeypatch):
    async def fake_load_account(saved_id):
        return _account(kill_engaged=False)
    monkeypatch.setattr(sfp_mod, "_load_account", fake_load_account)

    async def fake_get_setting_bool(key, default):
        return key == sfp_mod.PAPER_TRADING_KILL_SWITCH_KEY
    monkeypatch.setattr(sfp_mod, "get_setting_bool", fake_get_setting_bool)
    _patch_journal_helpers(monkeypatch)

    result = asyncio.run(sfp_mod.run_forward_paper_for_strategy(10))
    assert result["status"] == "skipped" and result["reason"] == "kill_switch"


# ---------------------------------------------------------------- _submit_one (checks reuse, submission, reconciliation)

def _decision_info(action="enter", reason="entry_rule", held_row=None, last_close=100.0, last_atr=2.0):
    return {
        "decision": {"action": action, "reason": reason},
        "last_close": last_close, "last_atr": last_atr, "held_row": held_row,
    }


def test_submit_one_blocks_a_buy_that_fails_the_order_value_limit_check(monkeypatch):
    journaled = _patch_journal_helpers(monkeypatch)

    # get_alpaca_latest_price is a SYNC function, called via run_in_threadpool
    # in production -- the fake must be sync too, or run_in_threadpool just
    # hands back an unawaited coroutine.
    monkeypatch.setattr(sfp_mod, "get_alpaca_latest_price", lambda ticker: 100.0)

    submit_calls = []

    def fake_submit_order(key, secret, **kw):
        submit_calls.append(kw)
        return {"id": "order-1"}
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "submit_order", fake_submit_order)

    async def fake_upsert(*a, **kw):
        pass
    monkeypatch.setattr(sfp_mod, "_upsert_position", fake_upsert)

    info = _decision_info(action="enter")
    # equity huge, weight 1.0 -> target_value way over a tiny max_order_value -> order_value_limit check fails
    result = asyncio.run(sfp_mod._submit_one(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, ticker="AAPL", side="buy",
        info=info, weights={"AAPL": 1.0}, equity=100_000.0, cooldown=5,
        max_order_value=10.0, max_pct=100.0, max_orders_per_day=100, collar_pct=100.0,
        restricted_symbols=set(), orders_today_count=0,
    ))
    assert result is None
    assert not submit_calls  # never reaches the broker
    assert any(evt == "rejected" for (_kind, evt, *_rest) in journaled if _kind == "journal")


def test_submit_one_submits_a_valid_buy_and_records_the_new_position(monkeypatch):
    _patch_journal_helpers(monkeypatch)
    monkeypatch.setattr(sfp_mod, "get_alpaca_latest_price", lambda ticker: 100.0)

    def fake_submit_order(key, secret, **kw):
        return {"id": "order-1"}
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "submit_order", fake_submit_order)

    upserts = []

    async def fake_upsert(account_id, ticker, **kw):
        upserts.append((ticker, kw))
    monkeypatch.setattr(sfp_mod, "_upsert_position", fake_upsert)

    info = _decision_info(action="enter", last_close=100.0, last_atr=3.0)
    result = asyncio.run(sfp_mod._submit_one(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, ticker="AAPL", side="buy",
        info=info, weights={"AAPL": 1.0}, equity=10_000.0, cooldown=5,
        max_order_value=1_000_000.0, max_pct=100.0, max_orders_per_day=100, collar_pct=100.0,
        restricted_symbols=set(), orders_today_count=0,
    ))
    assert result == {"ticker": "AAPL", "side": "buy", "alpaca_order_id": "order-1"}
    assert len(upserts) == 1
    ticker, kw = upserts[0]
    assert ticker == "AAPL"
    assert kw["qty"] == 100.0  # floor(10_000 / 100)
    assert kw["entry_price"] == 100.0
    assert kw["entry_atr"] == 3.0
    assert kw["cooldown_until_date"] is None


def test_submit_one_clears_the_position_and_starts_cooldown_on_a_sell(monkeypatch):
    _patch_journal_helpers(monkeypatch)
    monkeypatch.setattr(sfp_mod, "get_alpaca_latest_price", lambda ticker: (_ for _ in ()).throw(Exception("no live price")))

    def fake_submit_order(key, secret, **kw):
        return {"id": "order-2"}
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "submit_order", fake_submit_order)

    upserts = []

    async def fake_upsert(account_id, ticker, **kw):
        upserts.append((ticker, kw))
    monkeypatch.setattr(sfp_mod, "_upsert_position", fake_upsert)

    held_row = {"qty": 50.0, "entry_price": 90.0, "entry_atr": 2.0, "peak_price": 95.0}
    info = _decision_info(action="exit", reason="stop_loss", held_row=held_row, last_close=88.0)
    result = asyncio.run(sfp_mod._submit_one(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, ticker="AAPL", side="sell",
        info=info, weights={}, equity=10_000.0, cooldown=5,
        max_order_value=1_000_000.0, max_pct=100.0, max_orders_per_day=100, collar_pct=100.0,
        restricted_symbols=set(), orders_today_count=0,
    ))
    assert result == {"ticker": "AAPL", "side": "sell", "alpaca_order_id": "order-2"}
    ticker, kw = upserts[0]
    assert kw["qty"] == 0.0
    assert kw["entry_price"] is None
    assert kw["cooldown_until_date"] is not None


def test_submit_one_reconciles_an_ambiguous_network_failure_without_retrying(monkeypatch):
    _patch_journal_helpers(monkeypatch)
    monkeypatch.setattr(sfp_mod, "get_alpaca_latest_price", lambda ticker: 100.0)

    submit_attempts = []

    def fake_submit_order(key, secret, **kw):
        submit_attempts.append(kw)
        raise httpx.ConnectTimeout("network down")
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "submit_order", fake_submit_order)

    def fake_get_by_client_order_id(key, secret, client_order_id):
        return {"id": "order-recovered"}
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "get_order_by_client_order_id", fake_get_by_client_order_id)

    async def fake_upsert(*a, **kw):
        pass
    monkeypatch.setattr(sfp_mod, "_upsert_position", fake_upsert)

    info = _decision_info(action="enter")
    result = asyncio.run(sfp_mod._submit_one(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, ticker="AAPL", side="buy",
        info=info, weights={"AAPL": 1.0}, equity=10_000.0, cooldown=5,
        max_order_value=1_000_000.0, max_pct=100.0, max_orders_per_day=100, collar_pct=100.0,
        restricted_symbols=set(), orders_today_count=0,
    ))
    assert len(submit_attempts) == 1  # never blindly retried
    assert result == {"ticker": "AAPL", "side": "buy", "alpaca_order_id": "order-recovered"}


def test_submit_one_rejects_when_the_order_is_genuinely_refused(monkeypatch):
    _patch_journal_helpers(monkeypatch)
    monkeypatch.setattr(sfp_mod, "get_alpaca_latest_price", lambda ticker: 100.0)

    def fake_submit_order(key, secret, **kw):
        raise AlpacaTradingError(422, "insufficient buying power")
    monkeypatch.setattr(sfp_mod.alpaca_trading_client, "submit_order", fake_submit_order)

    info = _decision_info(action="enter")
    result = asyncio.run(sfp_mod._submit_one(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, ticker="AAPL", side="buy",
        info=info, weights={"AAPL": 1.0}, equity=10_000.0, cooldown=5,
        max_order_value=1_000_000.0, max_pct=100.0, max_orders_per_day=100, collar_pct=100.0,
        restricted_symbols=set(), orders_today_count=0,
    ))
    assert result is None


# ---------------------------------------------------------------- _submit_decisions (sells before buys)

def test_submit_decisions_submits_every_sell_before_any_buy(monkeypatch):
    _patch_journal_helpers(monkeypatch)

    async def fake_get_setting_float(key, default):
        return default
    async def fake_get_setting_int(key, default):
        return default
    async def fake_get_setting_str(key, default):
        return default
    monkeypatch.setattr(sfp_mod, "get_setting_float", fake_get_setting_float)
    monkeypatch.setattr(sfp_mod, "get_setting_int", fake_get_setting_int)
    monkeypatch.setattr(sfp_mod, "get_setting_str", fake_get_setting_str)

    async def fake_count_orders_today(account_id):
        return 0
    monkeypatch.setattr(sfp_mod, "_count_orders_today", fake_count_orders_today)

    order_calls = []

    async def fake_submit_one(run_id, account, creds, ticker, side, info, weights, equity, cooldown,
                              max_order_value, max_pct, max_orders_per_day, collar_pct, restricted_symbols,
                              orders_today_count):
        order_calls.append((ticker, side))
        return {"ticker": ticker, "side": side, "alpaca_order_id": f"order-{ticker}"}
    monkeypatch.setattr(sfp_mod, "_submit_one", fake_submit_one)

    decisions = {
        "AAPL": _decision_info(action="enter"),
        "MSFT": _decision_info(action="exit", held_row={"qty": 10.0}),
    }
    submitted = asyncio.run(sfp_mod._submit_decisions(
        run_id=1, account=_account(), creds={"key": "k", "secret": "s"}, decisions=decisions,
        weights={"AAPL": 1.0}, equity=10_000.0, cooldown=5,
    ))
    assert len(submitted) == 2
    sides_in_order = [side for _ticker, side in order_calls]
    assert sides_in_order == ["sell", "buy"]


# ---------------------------------------------------------------- get_track_record (reuses compute_member_performance)

def test_get_track_record_unavailable_when_no_account_is_linked(monkeypatch):
    class _FakeConn:
        async def fetchrow(self, sql, *a):
            return None

    class _FakeConnCtx:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(sfp_mod, "service_conn", lambda: _FakeConnCtx())
    result = asyncio.run(sfp_mod.get_track_record(10, "u1"))
    assert result == {"available": False, "reason": "No paper account linked for this strategy."}


def test_get_track_record_passes_snapshots_through_to_compute_member_performance(monkeypatch):
    from datetime import date, datetime

    account_row = {"id": 1, "created_at": datetime(2026, 1, 1)}
    snapshot_rows = [
        {"as_of_date": date(2026, 1, 2), "equity": 10_100.0},
        {"as_of_date": date(2026, 1, 3), "equity": 10_050.0},
    ]

    class _FakeConn:
        async def fetchrow(self, sql, *a):
            return account_row

        async def fetch(self, sql, *a):
            return snapshot_rows

    class _FakeConnCtx:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(sfp_mod, "service_conn", lambda: _FakeConnCtx())

    captured = {}

    def fake_compute(snapshots, start_date, end_date):
        captured["snapshots"] = snapshots
        captured["start_date"] = start_date
        return {"return_pct": -0.5, "days_of_data": 2}

    monkeypatch.setattr(sfp_mod, "compute_member_performance", fake_compute)

    result = asyncio.run(sfp_mod.get_track_record(10, "u1"))
    assert result["available"] is True
    assert result["return_pct"] == -0.5
    assert captured["start_date"] == date(2026, 1, 1)
    assert len(captured["snapshots"]) == 2
    assert len(result["equity_series"]) == 2
