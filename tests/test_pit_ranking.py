from datetime import date, timedelta

from services.pit_signal_service import compute_momentum_ranking_from_pit


def _rows(ticker, start_price, daily_pct, days=35, start_date=date(2026, 1, 1)):
    rows = []
    price = start_price
    for i in range(days):
        rows.append({"ticker": ticker, "price_date": start_date + timedelta(days=i), "close": price})
        price *= 1 + daily_pct
    return rows


def test_ranks_by_trailing_return_descending():
    rows = _rows("AAA", 100, 0.01) + _rows("BBB", 100, 0.005) + _rows("CCC", 100, -0.01)
    ranked = compute_momentum_ranking_from_pit(rows, lookback_days=30, top_n=3)
    assert [r["ticker"] for r in ranked] == ["AAA", "BBB", "CCC"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def test_trailing_return_value_matches_closed_form():
    rows = _rows("AAA", 100, 0.01)
    ranked = compute_momentum_ranking_from_pit(rows, lookback_days=30, top_n=1)
    expected = (100 * 1.01 ** 30 / 100 - 1) * 100
    assert abs(ranked[0]["trailing_return_pct"] - round(expected, 4)) < 1e-6


def test_insufficient_history_ticker_is_excluded_not_guessed_at():
    """A ticker with fewer than lookback_days + 1 PIT rows must be dropped,
    not ranked on partial data — the PIT store can't backfill from before
    capture began, so "not enough history yet" has to be an honest, visible
    exclusion rather than a silently wrong number."""
    enough = _rows("AAA", 100, 0.01)
    not_enough = _rows("DDD", 100, 0.05, days=10)
    ranked = compute_momentum_ranking_from_pit(enough + not_enough, lookback_days=30, top_n=5)
    tickers = [r["ticker"] for r in ranked]
    assert "DDD" not in tickers
    assert "AAA" in tickers


def test_top_n_truncates():
    rows = []
    for i in range(10):
        rows += _rows(f"T{i}", 100, 0.001 * i)
    ranked = compute_momentum_ranking_from_pit(rows, lookback_days=30, top_n=3)
    assert len(ranked) == 3


def test_empty_input_returns_empty():
    assert compute_momentum_ranking_from_pit([], lookback_days=30, top_n=25) == []


def test_zero_start_price_is_skipped_not_divide_by_zero():
    rows = _rows("ZERO", 0, 0.0)
    ranked = compute_momentum_ranking_from_pit(rows, lookback_days=30, top_n=5)
    assert ranked == []
