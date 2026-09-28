from datetime import datetime, timedelta, timezone

from web.backend.paper_order_sync import _alpaca_positions_to_df, reconcile_order_status


def _local(status="OPEN", filled_qty=0.0, filled_avg_price=None, alpaca_order_id="broker-1", created_at=None):
    return {
        "status": status, "filled_qty": filled_qty, "filled_avg_price": filled_avg_price,
        "alpaca_order_id": alpaca_order_id, "created_at": created_at or datetime.now(timezone.utc),
    }


def test_partial_fill_updates_filled_qty_and_avg_price_without_closing_order():
    local = _local(status="OPEN", filled_qty=0.0, filled_avg_price=None)
    broker_order = {"id": "broker-1", "status": "partially_filled", "filled_qty": "40", "filled_avg_price": "101.5"}
    update = reconcile_order_status(local, broker_order, datetime.now(timezone.utc))
    assert update == {
        "status": "PARTIALLY_FILLED", "filled_qty": 40.0, "filled_avg_price": 101.5,
        "reject_reason": None, "alpaca_order_id": "broker-1",
    }


def test_full_fill_sets_status_filled():
    local = _local(status="PARTIALLY_FILLED", filled_qty=40.0, filled_avg_price=101.5)
    broker_order = {"id": "broker-1", "status": "filled", "filled_qty": "100", "filled_avg_price": "101.75"}
    update = reconcile_order_status(local, broker_order, datetime.now(timezone.utc))
    assert update["status"] == "FILLED"
    assert update["filled_qty"] == 100.0


def test_rejection_sets_status_rejected_with_reason():
    local = _local(status="SUBMITTING", filled_qty=0.0, filled_avg_price=None)
    broker_order = {"id": "broker-1", "status": "rejected", "filled_qty": "0", "rejected_reason": "insufficient buying power"}
    update = reconcile_order_status(local, broker_order, datetime.now(timezone.utc))
    assert update["status"] == "REJECTED"
    assert update["reject_reason"] == "insufficient buying power"


def test_no_change_returns_none():
    local = _local(status="OPEN", filled_qty=0.0, filled_avg_price=None)
    broker_order = {"id": "broker-1", "status": "accepted", "filled_qty": "0"}
    assert reconcile_order_status(local, broker_order, datetime.now(timezone.utc)) is None


def test_order_missing_from_broker_before_grace_period_stays_unresolved():
    now = datetime.now(timezone.utc)
    local = _local(status="SUBMITTING", alpaca_order_id=None, created_at=now - timedelta(seconds=10))
    assert reconcile_order_status(local, None, now) is None


def test_order_missing_from_broker_after_grace_period_marked_unknown():
    now = datetime.now(timezone.utc)
    local = _local(status="SUBMITTING", alpaca_order_id=None, created_at=now - timedelta(seconds=120))
    update = reconcile_order_status(local, None, now)
    assert update["status"] == "UNKNOWN"


def test_alpaca_positions_to_df_maps_symbol_qty_avg_entry_price():
    positions = [
        {"symbol": "aapl", "qty": "10", "avg_entry_price": "150.0"},
        {"symbol": "MSFT", "qty": "5", "avg_entry_price": "410.25"},
    ]
    df = _alpaca_positions_to_df(positions)
    assert list(df["Ticker"]) == ["AAPL", "MSFT"]
    assert list(df["Shares"]) == [10.0, 5.0]
    assert list(df["Avg_Cost"]) == [150.0, 410.25]


def test_alpaca_positions_to_df_drops_non_positive_qty():
    positions = [{"symbol": "AAPL", "qty": "0", "avg_entry_price": "150.0"}]
    df = _alpaca_positions_to_df(positions)
    assert df.empty


def test_alpaca_positions_to_df_empty_returns_empty_with_columns():
    df = _alpaca_positions_to_df([])
    assert df.empty
    assert list(df.columns) == ["Ticker", "Shares", "Avg_Cost"]
