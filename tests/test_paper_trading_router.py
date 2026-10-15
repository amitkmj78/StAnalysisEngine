"""web/backend/routers/paper_trading.py: OrderRequest's per-order-type
field validation (BRK-2), and the new GET /performance endpoint's
assembly of compute_member_performance + SPY + model-portfolio (BRK-5/
PPR-1). Same fake-connection pattern as tests/test_admin_roles_and_theme.py.
"""

import asyncio
from datetime import date, datetime, timezone

from starlette.requests import Request

import web.backend.routers.paper_trading as router_mod
from web.backend.routers.paper_trading import OrderRequest, get_paper_trading_performance


def _request(user_id="u1") -> Request:
    scope = {
        "type": "http", "method": "GET", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    request = Request(scope)
    request.state.user = {"id": user_id}
    return request


class _FakeConn:
    def __init__(self, fetchrow=None, fetch=None):
        self._fetchrow = fetchrow or (lambda sql, *a: None)
        self._fetch = fetch or (lambda sql, *a: [])

    async def fetchrow(self, sql, *args):
        return self._fetchrow(sql, *args)

    async def fetch(self, sql, *args):
        return self._fetch(sql, *args)


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


# ---------------------------------------------------------------- OrderRequest.validate_fields_for_type

def _order(**overrides):
    base = dict(ticker="AAPL", side="buy", order_type="market", time_in_force="day", qty=10)
    base.update(overrides)
    return OrderRequest(**base)


def test_limit_order_requires_limit_price():
    assert _order(order_type="limit").validate_fields_for_type() is not None
    assert _order(order_type="limit", limit_price=100.0).validate_fields_for_type() is None


def test_stop_order_requires_stop_price():
    assert _order(order_type="stop").validate_fields_for_type() is not None
    assert _order(order_type="stop", stop_price=95.0).validate_fields_for_type() is None


def test_stop_limit_order_requires_both_prices():
    assert _order(order_type="stop_limit", stop_price=95.0).validate_fields_for_type() is not None
    assert _order(order_type="stop_limit", limit_price=94.0).validate_fields_for_type() is not None
    assert _order(order_type="stop_limit", stop_price=95.0, limit_price=94.0).validate_fields_for_type() is None


def test_trailing_stop_requires_exactly_one_of_trail_price_or_trail_percent():
    assert _order(order_type="trailing_stop").validate_fields_for_type() is not None
    assert _order(order_type="trailing_stop", trail_percent=5.0).validate_fields_for_type() is None
    assert _order(order_type="trailing_stop", trail_price=5.0).validate_fields_for_type() is None
    assert _order(order_type="trailing_stop", trail_price=5.0, trail_percent=5.0).validate_fields_for_type() is not None


def test_bracket_requires_both_take_profit_and_stop_loss():
    assert _order(order_class="bracket", take_profit_price=120.0).validate_fields_for_type() is not None
    assert _order(order_class="bracket", take_profit_price=120.0, stop_loss_price=90.0).validate_fields_for_type() is None


def test_market_order_needs_nothing_extra():
    assert _order().validate_fields_for_type() is None


# ---------------------------------------------------------------- GET /performance

def test_performance_assembles_account_spy_and_model_portfolio(monkeypatch):
    account_row = {"id": 1, "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    snapshot_rows = [
        {"as_of_date": date(2026, 1, 2), "equity": 10_100.0},
        {"as_of_date": date(2026, 1, 3), "equity": 10_050.0},
    ]
    conn = _FakeConn(fetch=lambda sql, *a: snapshot_rows if "paper_account_equity_snapshots" in sql else [])
    monkeypatch.setattr(router_mod, "user_conn", lambda user_id: _FakeConnCtx(conn))

    async def fake_get_account_or_404(conn, user_id, portfolio_id):
        return account_row
    monkeypatch.setattr(router_mod, "_get_account_or_404", fake_get_account_or_404)

    service_conn = _FakeConn(fetch=lambda sql, *a: [])
    monkeypatch.setattr(router_mod, "service_conn", lambda: _FakeConnCtx(service_conn))

    captured = {}

    def fake_compute_member_performance(snapshots, start_date, end_date):
        captured["snapshots"] = snapshots
        return {"return_pct": -0.5, "days_of_data": 2}
    monkeypatch.setattr(router_mod, "compute_member_performance", fake_compute_member_performance)

    async def fake_spy_return_pct(start_date, end_date):
        return 1.2
    monkeypatch.setattr(router_mod, "_spy_return_pct", fake_spy_return_pct)

    monkeypatch.setattr(router_mod, "build_model_portfolio_series", lambda rows, horizon: [["2026-01-01", 10000.0]])
    monkeypatch.setattr(router_mod, "model_portfolio_summary", lambda series: {"available": False, "months_of_history": 0.0})

    result = asyncio.run(get_paper_trading_performance(_request(), None))
    assert result["available"] is True
    assert result["account"] == {"return_pct": -0.5, "days_of_data": 2}
    assert result["spy_return_pct"] == 1.2
    assert result["model_portfolio"] == {"available": False, "months_of_history": 0.0}
    assert len(captured["snapshots"]) == 2
