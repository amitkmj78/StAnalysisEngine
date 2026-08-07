"""
TR-3's acceptance criterion: "an automated validation job fails the build
if any signal references data whose knowledge date is later than that
signal's as_of date." filter_knowledge_cutoff is the application-code half
of that guarantee (the other half is the SQL WHERE clause in
web/backend/pit_signals.fetch_pit_prices_as_of) — this test suite is what
makes that guarantee non-bypassable: it runs on every build via CI, not
just when someone remembers to check.
"""

from datetime import date, datetime, timezone

from services.pit_signal_service import filter_knowledge_cutoff


def _row(ticker, day, captured_at):
    return {"ticker": ticker, "price_date": date(2026, 1, day), "close": 1.0, "captured_at_utc": captured_at}


def test_row_captured_after_cutoff_is_excluded():
    cutoff = datetime(2026, 1, 20, tzinfo=timezone.utc)
    rows = [
        _row("X", 10, datetime(2026, 1, 10, tzinfo=timezone.utc)),
        _row("X", 25, datetime(2026, 1, 25, tzinfo=timezone.utc)),
    ]
    safe = filter_knowledge_cutoff(rows, cutoff)
    assert len(safe) == 1
    assert safe[0]["price_date"] == date(2026, 1, 10)


def test_row_captured_exactly_at_cutoff_is_kept():
    cutoff = datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc)
    rows = [_row("X", 20, cutoff)]
    assert len(filter_knowledge_cutoff(rows, cutoff)) == 1


def test_naive_datetime_is_treated_as_utc_not_silently_dropped_or_kept_wrong():
    cutoff = datetime(2026, 1, 20, tzinfo=timezone.utc)
    past_naive = _row("X", 10, datetime(2026, 1, 10))  # no tzinfo
    future_naive = _row("Y", 25, datetime(2026, 1, 25))  # no tzinfo
    safe = filter_knowledge_cutoff([past_naive, future_naive], cutoff)
    assert [r["ticker"] for r in safe] == ["X"]


def test_no_future_row_survives_regardless_of_input_order():
    cutoff = datetime(2026, 1, 15, tzinfo=timezone.utc)
    rows = [_row("X", d, datetime(2026, 1, d, tzinfo=timezone.utc)) for d in range(30, 0, -1)]
    safe = filter_knowledge_cutoff(rows, cutoff)
    assert all(r["captured_at_utc"] <= cutoff for r in safe)
    assert len(safe) == 15


def test_empty_input_is_safe():
    cutoff = datetime(2026, 1, 20, tzinfo=timezone.utc)
    assert filter_knowledge_cutoff([], cutoff) == []
