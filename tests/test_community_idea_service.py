from datetime import date

import pandas as pd
import pytest

from services.community_idea_service import (
    MIN_IDEAS_FOR_LEADERBOARD,
    evaluate_idea_outcome,
    leaderboard_sort_key,
    risk_adjusted_excess_return,
    worst_idea,
)


def _closes(prices, start="2026-01-01"):
    return pd.Series(prices, index=pd.bdate_range(start, periods=len(prices)))


# --- evaluate_idea_outcome: the LONG/SHORT -> Buy/Trim adapter ---


def test_long_idea_hit_when_price_rises():
    closes = _closes([100, 101, 102, 103])
    result = evaluate_idea_outcome(date(2026, 1, 1), "LONG", closes, horizon_days=3)
    assert result["outcome"] == "hit"
    assert result["realized_return_pct"] > 0


def test_long_idea_miss_when_price_falls():
    closes = _closes([100, 99, 98, 97])
    result = evaluate_idea_outcome(date(2026, 1, 1), "LONG", closes, horizon_days=3)
    assert result["outcome"] == "miss"


def test_short_idea_hit_when_price_falls():
    closes = _closes([100, 99, 98, 97])
    result = evaluate_idea_outcome(date(2026, 1, 1), "SHORT", closes, horizon_days=3)
    assert result["outcome"] == "hit"


def test_short_idea_miss_when_price_rises():
    closes = _closes([100, 101, 102, 103])
    result = evaluate_idea_outcome(date(2026, 1, 1), "SHORT", closes, horizon_days=3)
    assert result["outcome"] == "miss"


def test_none_when_horizon_not_yet_elapsed():
    closes = _closes([100, 101])
    assert evaluate_idea_outcome(date(2026, 1, 1), "LONG", closes, horizon_days=5) is None


def test_invalid_direction_raises():
    closes = _closes([100, 101, 102])
    with pytest.raises(ValueError):
        evaluate_idea_outcome(date(2026, 1, 1), "SIDEWAYS", closes, horizon_days=1)


# --- risk_adjusted_excess_return: COM-4's ranking ---


def test_none_below_minimum_sample():
    returns = [1.0] * (MIN_IDEAS_FOR_LEADERBOARD - 1)
    assert risk_adjusted_excess_return(returns) is None


def test_real_score_at_minimum_sample():
    returns = [2.0, -1.0, 3.0, 0.5, 1.5, -0.5, 2.5, 1.0, 0.0, 1.0]
    assert len(returns) == MIN_IDEAS_FOR_LEADERBOARD
    score = risk_adjusted_excess_return(returns)
    assert score is not None
    # The real function rounds to 4 decimals -- compare at that precision.
    assert score == pytest.approx(statistics_mean_over_stdev(returns), abs=1e-4)


def statistics_mean_over_stdev(values):
    import statistics
    return statistics.mean(values) / statistics.pstdev(values)


def test_none_not_zero_division_when_all_returns_identical():
    returns = [1.0] * MIN_IDEAS_FOR_LEADERBOARD
    assert risk_adjusted_excess_return(returns) is None


def test_higher_mean_same_spread_scores_higher():
    low = [1.0, -1.0] * (MIN_IDEAS_FOR_LEADERBOARD // 2)
    high = [2.0, 0.0] * (MIN_IDEAS_FOR_LEADERBOARD // 2)
    assert risk_adjusted_excess_return(high) > risk_adjusted_excess_return(low)


# --- leaderboard_sort_key: None always sorts last ---


def test_sort_key_puts_none_last():
    entries = [{"score": 5.0}, {"score": None}, {"score": 10.0}]
    ranked = sorted(entries, key=lambda e: leaderboard_sort_key(e["score"]))
    assert [e["score"] for e in ranked] == [10.0, 5.0, None]


# --- worst_idea: per-direction worst-miss convention ---


def test_worst_idea_among_longs_is_the_biggest_loss():
    ideas = [
        {"direction": "LONG", "realized_return_pct": 5.0},
        {"direction": "LONG", "realized_return_pct": -20.0},
        {"direction": "LONG", "realized_return_pct": -2.0},
    ]
    assert worst_idea(ideas)["realized_return_pct"] == -20.0


def test_worst_idea_among_shorts_is_the_biggest_gain_against_them():
    ideas = [
        {"direction": "SHORT", "realized_return_pct": -5.0},  # a SHORT hit (price fell) -- not a miss
        {"direction": "SHORT", "realized_return_pct": 15.0},  # price rose hard against the SHORT -- the real miss
    ]
    assert worst_idea(ideas)["realized_return_pct"] == 15.0


def test_worst_idea_mixed_directions_picks_the_true_worst():
    ideas = [
        {"direction": "LONG", "realized_return_pct": -3.0},
        {"direction": "SHORT", "realized_return_pct": 25.0},  # worst: a SHORT that rose 25%
    ]
    assert worst_idea(ideas)["realized_return_pct"] == 25.0
    assert worst_idea(ideas)["direction"] == "SHORT"


def test_worst_idea_none_for_empty_list():
    assert worst_idea([]) is None
