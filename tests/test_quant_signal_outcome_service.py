from datetime import date

from services.quant_signal_outcome_service import (
    compute_quant_signal_outcomes,
    summarize_quant_signal_outcomes,
)


def _prices(ticker: str, start: date, closes: list[float]) -> list[dict]:
    from datetime import timedelta

    return [
        {"ticker": ticker, "price_date": start + timedelta(days=i), "close": c}
        for i, c in enumerate(closes)
    ]


def test_buy_marked_correct_when_price_rises_by_horizon():
    signal_rows = [
        {"ticker": "AAA", "as_of_date": date(2026, 1, 1), "signal": "BUY", "expected_return_pct": 6.0, "last_close": 100.0}
    ]
    # 5 trading days after as_of_date (index 5), price is up.
    price_rows = _prices("AAA", date(2026, 1, 1), [100.0, 101, 102, 103, 104, 110.0])

    outcomes = compute_quant_signal_outcomes(signal_rows, price_rows, horizon_days=5)

    assert len(outcomes) == 1
    o = outcomes[0]
    assert o["exit_price"] == 110.0
    assert o["realized_return_pct"] == 10.0
    assert o["correct"] is True


def test_sell_marked_correct_when_price_falls():
    signal_rows = [
        {"ticker": "BBB", "as_of_date": date(2026, 1, 1), "signal": "SELL", "expected_return_pct": -6.0, "last_close": 100.0}
    ]
    price_rows = _prices("BBB", date(2026, 1, 1), [100.0, 99, 98, 97, 96, 90.0])

    outcomes = compute_quant_signal_outcomes(signal_rows, price_rows, horizon_days=5)

    assert outcomes[0]["correct"] is True


def test_hold_correct_only_within_the_band():
    inside = [
        {"ticker": "CCC", "as_of_date": date(2026, 1, 1), "signal": "HOLD", "expected_return_pct": 1.0, "last_close": 100.0}
    ]
    price_rows_inside = _prices("CCC", date(2026, 1, 1), [100.0, 100, 100, 100, 100, 102.0])
    outcomes_inside = compute_quant_signal_outcomes(inside, price_rows_inside, horizon_days=5)
    assert outcomes_inside[0]["correct"] is True

    outside = [
        {"ticker": "DDD", "as_of_date": date(2026, 1, 1), "signal": "HOLD", "expected_return_pct": 1.0, "last_close": 100.0}
    ]
    price_rows_outside = _prices("DDD", date(2026, 1, 1), [100.0, 100, 100, 100, 100, 108.0])
    outcomes_outside = compute_quant_signal_outcomes(outside, price_rows_outside, horizon_days=5)
    assert outcomes_outside[0]["correct"] is False


def test_not_due_yet_when_fewer_than_horizon_days_of_prices_exist():
    signal_rows = [
        {"ticker": "AAA", "as_of_date": date(2026, 1, 1), "signal": "BUY", "expected_return_pct": 6.0, "last_close": 100.0}
    ]
    # Only 3 trading days captured after as_of_date -- horizon_days=5 not reached.
    price_rows = _prices("AAA", date(2026, 1, 1), [100.0, 101, 102, 103])

    outcomes = compute_quant_signal_outcomes(signal_rows, price_rows, horizon_days=5)
    assert outcomes == []


def test_skips_signal_with_no_matching_entry_price_row():
    signal_rows = [
        {"ticker": "ZZZ", "as_of_date": date(2026, 1, 1), "signal": "BUY", "expected_return_pct": 6.0, "last_close": 100.0}
    ]
    # ZZZ has price history, but none on the exact as_of_date.
    price_rows = _prices("ZZZ", date(2026, 1, 2), [101, 102, 103, 104, 105, 106])

    outcomes = compute_quant_signal_outcomes(signal_rows, price_rows, horizon_days=5)
    assert outcomes == []


def test_summarize_pools_win_rate_by_signal():
    outcomes = [
        {"ticker": "A", "as_of_date": date(2026, 1, 1), "signal": "BUY", "expected_return_pct": 6.0,
         "entry_price": 100.0, "horizon_days": 10, "exit_date": date(2026, 1, 15), "exit_price": 110.0,
         "realized_return_pct": 10.0, "correct": True},
        {"ticker": "B", "as_of_date": date(2026, 1, 1), "signal": "BUY", "expected_return_pct": 6.0,
         "entry_price": 100.0, "horizon_days": 10, "exit_date": date(2026, 1, 15), "exit_price": 95.0,
         "realized_return_pct": -5.0, "correct": False},
        {"ticker": "C", "as_of_date": date(2026, 1, 1), "signal": "SELL", "expected_return_pct": -6.0,
         "entry_price": 100.0, "horizon_days": 10, "exit_date": date(2026, 1, 15), "exit_price": 95.0,
         "realized_return_pct": -5.0, "correct": True},
    ]

    summary = summarize_quant_signal_outcomes(outcomes)

    assert summary["BUY"] == {"count": 2, "win_rate_pct": 50.0}
    assert summary["SELL"] == {"count": 1, "win_rate_pct": 100.0}
    assert summary["HOLD"] == {"count": 0, "win_rate_pct": None}
