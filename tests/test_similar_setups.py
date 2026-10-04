from datetime import date, timedelta

import pandas as pd
import pytest

from services.similar_setups import MIN_CASES, find_similar_setups


def _closes(n=40, start=100.0, step=1.0):
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.Series([start + step * i for i in range(n)], index=pd.to_datetime(dates))


def test_matches_only_days_within_the_score_band_and_the_same_regime():
    closes = _closes()
    rows = [
        {"date": date(2026, 1, 2), "short_score": 60.0, "regime": "Neutral"},   # match
        {"date": date(2026, 1, 3), "short_score": 64.0, "regime": "Neutral"},   # match (within 5)
        {"date": date(2026, 1, 4), "short_score": 70.0, "regime": "Neutral"},   # score too far
        {"date": date(2026, 1, 5), "short_score": 60.0, "regime": "Cautious"},  # regime differs
    ]
    r = find_similar_setups(rows, closes, current_score=62.0, current_regime="Neutral")
    assert r["n"] == 2


def test_return_is_measured_over_the_horizon_from_each_matching_close():
    closes = _closes(step=1.0)  # close on day i is 100 + i
    rows = [{"date": date(2026, 1, 2), "short_score": 60.0, "regime": "Neutral"}]  # index 1, close 101
    r = find_similar_setups(rows, closes, current_score=60.0, current_regime="Neutral", horizon=10)
    # close 10 sessions later is 111: 111/101 - 1 = 9.90099%
    assert r["median_return_pct"] == pytest.approx(9.90, abs=0.01)
    assert r["min_return_pct"] == r["max_return_pct"] == r["median_return_pct"]


def test_days_without_a_full_forward_window_are_left_out():
    closes = _closes(n=20)
    rows = [{"date": date(2026, 1, 18), "short_score": 60.0, "regime": "Neutral"}]  # only 1 session after
    r = find_similar_setups(rows, closes, current_score=60.0, current_regime="Neutral", horizon=10)
    assert r["n"] == 0
    assert r["median_return_pct"] is None


def test_under_thirty_cases_carries_the_caveat_and_thirty_or_more_does_not():
    closes = _closes(n=200)
    rows = [{"date": date(2026, 1, 1) + timedelta(days=i), "short_score": 60.0, "regime": "Neutral"} for i in range(40)]
    few = find_similar_setups(rows[:5], closes, 60.0, "Neutral")
    assert "too few" in few["caveat"]
    many = find_similar_setups(rows, closes, 60.0, "Neutral")
    assert many["n"] >= MIN_CASES
    assert many["caveat"] is None


def test_no_regime_label_matches_any_regime():
    closes = _closes()
    rows = [{"date": date(2026, 1, 2), "short_score": 60.0, "regime": "Risk-Off"}]
    r = find_similar_setups(rows, closes, 60.0, None)
    assert r["n"] == 1
