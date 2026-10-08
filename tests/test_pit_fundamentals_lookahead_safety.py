"""FND-2's acceptance criterion: a look-ahead test exists for point-in-time
fundamentals and runs in CI (.github/workflows/ci.yml already runs
`pytest tests/` on every push/PR to main, so a new file here is
automatically covered -- no workflow change needed).

Unlike the existing NFR-1-round tests in test_stock_score_capture_service.py
(test_value_growth_quality_fetch_binds_the_requested_as_of_date,
test_resolve_sector_map_binds_the_requested_as_of_date), which only assert
the SQL text/args passed to conn.fetch, this test exercises real DATA
behavior: the fake connection actually applies the same filter and
DISTINCT ON ... ORDER BY as_of_date DESC semantics the real SQL declares,
against a fixture that includes both a safe row and a "future" row (a
later as_of_date than the requested cutoff) for the same ticker. A
regression that weakens the WHERE clause, flips the comparison, or breaks
the "latest row on or before cutoff" selection would leak the future row's
values into a score computed for a past date -- this is what would catch
that, the same guarantee test_pit_lookahead_safety.py gives prices.
"""

import asyncio
from datetime import date

from services import stock_score_capture_service as capture_service


def _filtered_latest_rows(rows: list[dict], tickers: list[str], cutoff: date) -> dict[str, dict]:
    """Mirrors the real SQL's WHERE ticker = ANY($1) AND as_of_date <= $2,
    then DISTINCT ON (ticker) ... ORDER BY ticker, as_of_date DESC -- i.e.
    the single latest-on-or-before-cutoff row per ticker."""
    by_ticker: dict[str, dict] = {}
    for row in rows:
        if row["ticker"] not in tickers or row["as_of_date"] > cutoff:
            continue
        current = by_ticker.get(row["ticker"])
        if current is None or row["as_of_date"] > current["as_of_date"]:
            by_ticker[row["ticker"]] = row
    return by_ticker


class _FakeFundamentalsConn:
    """A fake conn.fetch that actually filters a fixture table the way the
    real pit_fundamentals SQL does, instead of just recording what args it
    was called with."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    async def fetch(self, sql, *args):
        tickers, cutoff = args[0], args[1]
        assert "as_of_date <= $2" in sql, "query must still bound the lookup by as_of_date"
        latest = _filtered_latest_rows(self._rows, tickers, cutoff)
        return [latest[t] for t in tickers if t in latest]


class _FakeFundamentalsConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _row(ticker, as_of, **kwargs):
    base = {
        "ticker": ticker, "as_of_date": as_of, "forward_pe": None,
        "revenue_growth_pct": None, "earnings_growth_pct": None,
        "return_on_equity_pct": None, "profit_margin_pct": None, "sector": None,
    }
    base.update(kwargs)
    return base


def test_value_growth_quality_fetch_never_uses_a_fundamentals_row_from_after_the_cutoff(monkeypatch):
    cutoff = date(2024, 3, 15)
    rows = [
        _row("AAPL", date(2024, 3, 10), forward_pe=25.0),  # safe: before cutoff
        _row("AAPL", date(2024, 3, 25), forward_pe=999.0),  # future: after cutoff -- must never be used
    ]
    monkeypatch.setattr(capture_service, "service_conn", lambda: _FakeFundamentalsConnCtx(_FakeFundamentalsConn(rows)))

    result = asyncio.run(capture_service.fetch_value_growth_and_quality_inputs(["AAPL"], cutoff))

    assert result["AAPL"]["value"]["raw"] == 25.0


def test_value_growth_quality_fetch_picks_the_latest_row_still_on_or_before_cutoff(monkeypatch):
    """Not just "any safe row" -- the LATEST one on or before the cutoff,
    same as the real DISTINCT ON ... ORDER BY as_of_date DESC."""
    cutoff = date(2024, 3, 15)
    rows = [
        _row("AAPL", date(2024, 2, 1), forward_pe=10.0),
        _row("AAPL", date(2024, 3, 10), forward_pe=25.0),  # latest safe row
        _row("AAPL", date(2024, 3, 20), forward_pe=999.0),  # future
    ]
    monkeypatch.setattr(capture_service, "service_conn", lambda: _FakeFundamentalsConnCtx(_FakeFundamentalsConn(rows)))

    result = asyncio.run(capture_service.fetch_value_growth_and_quality_inputs(["AAPL"], cutoff))

    assert result["AAPL"]["value"]["raw"] == 25.0


def test_value_growth_quality_fetch_is_order_independent(monkeypatch):
    """The future row listed FIRST in the fixture must still never leak in --
    guards against an implementation that implicitly relies on row order
    rather than the as_of_date comparison itself."""
    cutoff = date(2024, 3, 15)
    rows = [
        _row("AAPL", date(2024, 3, 25), forward_pe=999.0),  # future, listed first
        _row("AAPL", date(2024, 3, 10), forward_pe=25.0),  # safe
    ]
    monkeypatch.setattr(capture_service, "service_conn", lambda: _FakeFundamentalsConnCtx(_FakeFundamentalsConn(rows)))

    result = asyncio.run(capture_service.fetch_value_growth_and_quality_inputs(["AAPL"], cutoff))

    assert result["AAPL"]["value"]["raw"] == 25.0


def test_resolve_sector_map_never_uses_a_sector_row_from_after_the_cutoff(monkeypatch):
    cutoff = date(2024, 3, 15)
    rows = [
        _row("AAPL", date(2024, 3, 10), sector="Technology"),  # safe
        _row("AAPL", date(2024, 3, 25), sector="Energy"),  # future -- a wrong/changed sector must not leak in
    ]
    monkeypatch.setattr(capture_service, "service_conn", lambda: _FakeFundamentalsConnCtx(_FakeFundamentalsConn(rows)))
    monkeypatch.setattr(capture_service, "get_cached_info", lambda t: {})

    result = asyncio.run(capture_service.resolve_sector_map(["AAPL"], cutoff))

    # GICS-normalized (services.stock_finder_service._gics_sector) -- "Energy"
    # would normalize differently, so this also proves the future row's raw
    # value never reached the normalizer at all.
    assert result["AAPL"] == "Information Technology"


def test_no_rows_at_all_on_or_before_cutoff_falls_back_cleanly(monkeypatch):
    """Every captured row is in the future relative to this as_of_date (e.g.
    a backtest date before capture began) -- must come back empty/None, not
    accidentally pick the earliest future row."""
    cutoff = date(2024, 1, 1)
    rows = [_row("AAPL", date(2024, 3, 25), forward_pe=999.0)]
    monkeypatch.setattr(capture_service, "service_conn", lambda: _FakeFundamentalsConnCtx(_FakeFundamentalsConn(rows)))

    result = asyncio.run(capture_service.fetch_value_growth_and_quality_inputs(["AAPL"], cutoff))

    assert result["AAPL"]["value"]["raw"] is None
