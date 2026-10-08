"""FND-3's DB-orchestration layer: web.backend.signal_publication.
evaluate_due_stock_page_signal_outcomes. The pure evaluation math itself
(services.signal_publication_service.evaluate_stock_page_signal_outcomes)
is covered thoroughly in tests/test_signal_publication_service.py; this
exercises the full round trip -- reading due stock_scores rows, batching
the PIT price lookups, and inserting the result -- against a fake
connection, the same role a real dev-DB run plays for this plan's own
verification step but runnable in CI without one.
"""

import asyncio
from datetime import date

import web.backend.signal_publication as signal_publication


class _FakeConn:
    def __init__(self, due_rows, price_rows, spy_rows):
        self._due_rows = due_rows
        self._price_rows = price_rows
        self._spy_rows = spy_rows
        self.inserted: list[tuple] = []

    async def fetch(self, sql, *args):
        if "FROM stock_scores" in sql:
            return self._due_rows
        if "ticker = 'SPY'" in sql:
            return self._spy_rows
        if "FROM pit_prices" in sql:
            return self._price_rows
        raise AssertionError(f"Unexpected query: {sql}")

    async def execute(self, sql, *args):
        assert "INSERT INTO stock_signal_outcomes" in sql
        self.inserted.append(args)
        return "INSERT 0 1"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _due_row(ticker, as_of_date, signal="Buy", score=75.0, label="medium", weights_version="wv1"):
    return {
        "ticker": ticker, "as_of_date": as_of_date, "short_signal": signal,
        "short_confidence_score": score, "short_confidence_label": label, "weights_version": weights_version,
    }


def _price_row(ticker, price_date, close):
    return {"ticker": ticker, "price_date": price_date, "close": close}


def test_evaluate_due_stock_page_signal_outcomes_inserts_a_matured_buy():
    due_rows = [_due_row("AAPL", date(2026, 1, 1))]
    # 11 consecutive calendar days used as a stand-in trading calendar --
    # evaluate_signal_outcome only cares about index order/spacing, not
    # real weekday gaps, for this fake's purposes.
    price_rows = [_price_row("AAPL", date(2026, 1, 1 + i), 100.0 + i) for i in range(11)]
    spy_rows = [_price_row("SPY", date(2026, 1, 1 + i), 500.0) for i in range(11)]
    conn = _FakeConn(due_rows, price_rows, spy_rows)

    inserted = asyncio.run(_run(conn))

    assert inserted == 1
    assert len(conn.inserted) == 1
    args = conn.inserted[0]
    assert args[0] == "AAPL"
    assert args[3] == "Buy"


def test_evaluate_due_stock_page_signal_outcomes_returns_zero_when_nothing_due():
    conn = _FakeConn([], [], [])
    inserted = asyncio.run(_run(conn))
    assert inserted == 0
    assert conn.inserted == []


def test_evaluate_due_stock_page_signal_outcomes_skips_a_ticker_with_no_price_history():
    due_rows = [_due_row("AAPL", date(2026, 1, 1)), _due_row("ZZZZ", date(2026, 1, 1))]
    price_rows = [_price_row("AAPL", date(2026, 1, 1 + i), 100.0 + i) for i in range(11)]
    spy_rows = [_price_row("SPY", date(2026, 1, 1 + i), 500.0) for i in range(11)]
    conn = _FakeConn(due_rows, price_rows, spy_rows)  # no ZZZZ price rows at all

    inserted = asyncio.run(_run(conn))

    assert inserted == 1
    assert [a[0] for a in conn.inserted] == ["AAPL"]


async def _run(conn):
    import unittest.mock as mock

    with mock.patch.object(signal_publication, "service_conn", lambda: _FakeConnCtx(conn)):
        return await signal_publication.evaluate_due_stock_page_signal_outcomes(horizon_days=10)
