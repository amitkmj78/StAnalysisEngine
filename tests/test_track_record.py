import pandas as pd

from services.track_record import MIN_SIGNALS, track_record


def _series(values, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_too_few_signals_show_not_enough_data_and_no_figures():
    closes = _series([100 + i for i in range(60)])
    spy = _series([100.0] * 60)
    rows = [{"as_of_date": str(closes.index[i].date()), "short_signal": "Buy"} for i in range(5)]
    result = track_record(rows, closes, spy)
    assert result["enough_data"] is False
    assert result["hit_rate_pct"] is None
    assert "Not enough data yet" in result["message"]


def test_hit_rate_excess_return_and_worst_miss_on_enough_signals():
    # Prices rise steadily, so every Buy is a hit, and every Trim is a miss.
    n = 200
    closes = _series([100 + i for i in range(n)])
    spy = _series([100.0] * n)
    rows = []
    for i in range(0, 120):
        signal = "Buy" if i % 2 == 0 else "Trim"
        rows.append({"as_of_date": str(closes.index[i].date()), "short_signal": signal})
    result = track_record(rows, closes, spy)
    assert result["enough_data"] is True
    assert result["signals_evaluated"] >= MIN_SIGNALS
    # Each Buy rose, so it's a hit; each Trim rose, so it's a miss.
    assert 0 < result["hit_rate_pct"] < 100
    assert result["worst_miss"]["signal"] == "Trim"
    # SPY is flat, so the excess return equals the stock's move, which is positive for these prices.
    assert result["avg_excess_vs_spy_pct"] > 0


def test_hold_signals_are_left_out():
    n = 200
    closes = _series([100 + i for i in range(n)])
    spy = _series([100.0] * n)
    rows = [{"as_of_date": str(closes.index[i].date()), "short_signal": "Hold"} for i in range(120)]
    result = track_record(rows, closes, spy)
    assert result["signals_evaluated"] == 0
    assert result["enough_data"] is False


def test_excess_return_is_measured_against_spy_over_the_same_dates():
    n = 200
    closes = _series([100 + i for i in range(n)])
    spy = _series([100 + 0.5 * i for i in range(n)])
    rows = [{"as_of_date": str(closes.index[i].date()), "short_signal": "Buy"} for i in range(120)]
    result = track_record(rows, closes, spy)
    # The stock moves faster than SPY in every window, so the average excess is positive.
    assert result["avg_excess_vs_spy_pct"] > 0
