from datetime import date

import pandas as pd
import pytest

from services.condition_alert_service import (
    CATEGORY_FIELDS,
    NUMERIC_FIELDS,
    build_condition_frame,
    evaluate_condition_alert,
    is_intraday_eligible,
    parse_condition,
    parse_conditions,
)
from services.strategy_engine import REGIME_LABELS, Rule


def test_parse_condition_numeric_field():
    r = parse_condition({"field": "rsi_14", "op": "<", "value": "30"})
    assert r == Rule("rsi_14", "<", 30.0)


def test_parse_condition_price_field_is_new_vocabulary_not_in_strategy_engine():
    assert "price" in NUMERIC_FIELDS
    r = parse_condition({"field": "price", "op": ">", "value": "150"})
    assert r.field == "price"


def test_parse_condition_score_fields():
    assert parse_condition({"field": "short_score", "op": ">=", "value": "70"}).field == "short_score"
    assert parse_condition({"field": "long_score", "op": "<=", "value": "40"}).field == "long_score"


def test_parse_condition_signal_field_category_op():
    r = parse_condition({"field": "short_signal", "op": "is", "value": "Buy"})
    assert r == Rule("short_signal", "is", "Buy")


def test_parse_condition_signal_field_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_condition({"field": "short_signal", "op": "is", "value": "Strong Buy"})


def test_parse_condition_regime_field():
    assert "regime" in CATEGORY_FIELDS
    assert CATEGORY_FIELDS["regime"] == REGIME_LABELS
    r = parse_condition({"field": "regime", "op": "is", "value": "Risk-On"})
    assert r.value == "Risk-On"


def test_parse_condition_unknown_field_rejected():
    with pytest.raises(ValueError):
        parse_condition({"field": "nonsense", "op": ">", "value": "1"})


def test_parse_condition_numeric_field_rejects_category_op():
    with pytest.raises(ValueError):
        parse_condition({"field": "price", "op": "is", "value": "Buy"})


def test_parse_conditions_requires_at_least_one():
    with pytest.raises(ValueError):
        parse_conditions([])


def test_parse_conditions_caps_at_max():
    conditions = [{"field": "price", "op": ">", "value": str(i)} for i in range(10)]
    with pytest.raises(ValueError):
        parse_conditions(conditions)


# ---------------------------------------------------------------------------
# ALX-1's literal acceptance criterion: "a 3-condition alert saves and
# fires correctly in a replay test." Three conditions (price, score,
# regime) each become true on a different, independently-controlled day;
# combined with AND, the alert must fire starting on the LATEST of the
# three days, never before.
# ---------------------------------------------------------------------------


def _ohlc(prices: list[float], start="2024-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(prices))
    close = pd.Series(prices, index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close})


def _replay_fixture():
    n = 260
    prices = [100.0 + i * 0.3 for i in range(n)]  # price crosses 150 around day 167
    ohlc = _ohlc(prices)
    dates = ohlc.index

    # Regime flips to Risk-On on day 180, Neutral before.
    regime_by_date = {
        d.strftime("%Y-%m-%d"): ("Risk-On" if i >= 180 else "Neutral") for i, d in enumerate(dates)
    }
    # Score jumps to 80 on day 200, 50 before -- captured as two discrete
    # snapshot rows (not a continuous series), same as the real
    # stock_scores table.
    score_history = [
        {"as_of_date": dates[0].date(), "short_score": 50.0, "long_score": 50.0, "short_signal": "Hold", "long_signal": "Hold"},
        {"as_of_date": dates[200].date(), "short_score": 80.0, "long_score": 80.0, "short_signal": "Buy", "long_signal": "Buy"},
    ]
    conditions = parse_conditions([
        {"field": "price", "op": ">", "value": "150"},
        {"field": "regime", "op": "is", "value": "Risk-On"},
        {"field": "short_score", "op": ">=", "value": "75"},
    ])
    return ohlc, regime_by_date, score_history, conditions


def test_replay_three_condition_and_alert_fires_only_from_the_latest_qualifying_day():
    ohlc, regime_by_date, score_history, conditions = _replay_fixture()

    # Day 199 (0-indexed slice end): price and regime already qualify
    # (crossed at ~167 and 180), but the score snapshot captured on day
    # 200 hasn't happened yet in this replay slice -- must not fire.
    frame_before = build_condition_frame(ohlc.iloc[:200], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "AND", frame_before) is False

    # Day 200: the score snapshot is now in the replayed slice, and the
    # other two conditions are still true -- must fire for the first time.
    frame_at = build_condition_frame(ohlc.iloc[:201], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "AND", frame_at) is True

    # A day well before ANY condition qualifies (day 50: price ~115,
    # regime Neutral, score 50) must not fire.
    frame_early = build_condition_frame(ohlc.iloc[:51], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "AND", frame_early) is False


def test_replay_same_three_conditions_with_or_fires_much_earlier():
    # Same fixture, OR instead of AND: fires as soon as the FIRST
    # condition (price > 150, ~day 167) is true, not the last.
    ohlc, regime_by_date, score_history, conditions = _replay_fixture()

    frame_before_price = build_condition_frame(ohlc.iloc[:160], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "OR", frame_before_price) is False

    frame_after_price = build_condition_frame(ohlc.iloc[:200], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "OR", frame_after_price) is True


def test_evaluate_condition_alert_none_when_a_field_has_no_history_yet():
    ohlc, regime_by_date, _, _ = _replay_fixture()
    frame = build_condition_frame(ohlc.iloc[:60], regime_by_date, [], score_history=None)
    conditions = parse_conditions([{"field": "long_score", "op": ">", "value": "50"}])
    # No score_history at all -- long_score is an all-null column, so the
    # latest value is NaN, not knowable, not guessed at as False.
    assert evaluate_condition_alert(conditions, "AND", frame) is None


def test_evaluate_condition_alert_none_for_empty_frame():
    assert evaluate_condition_alert([], "AND", pd.DataFrame()) is None


def test_evaluate_condition_alert_none_for_invalid_combinator():
    ohlc, regime_by_date, score_history, conditions = _replay_fixture()
    frame = build_condition_frame(ohlc.iloc[:201], regime_by_date, [], score_history)
    assert evaluate_condition_alert(conditions, "XOR", frame) is None


def test_build_condition_frame_price_field_aliases_close():
    ohlc, regime_by_date, _, _ = _replay_fixture()
    frame = build_condition_frame(ohlc.iloc[:60], regime_by_date, [])
    assert (frame["price"] == frame["close"]).all()


# --- ALX-2: intraday eligibility (only price/indicator conditions can be
# usefully re-checked faster than once a day) ---


def test_is_intraday_eligible_true_for_price_and_indicator_conditions():
    conditions = parse_conditions([
        {"field": "price", "op": ">", "value": "150"},
        {"field": "rsi_14", "op": "<", "value": "30"},
    ])
    assert is_intraday_eligible(conditions) is True


def test_is_intraday_eligible_false_when_any_condition_is_daily_only():
    conditions = parse_conditions([
        {"field": "price", "op": ">", "value": "150"},
        {"field": "regime", "op": "is", "value": "Risk-On"},
    ])
    assert is_intraday_eligible(conditions) is False


def test_is_intraday_eligible_false_for_score_and_signal_fields():
    assert is_intraday_eligible(parse_conditions([{"field": "short_score", "op": ">", "value": "70"}])) is False
    assert is_intraday_eligible(parse_conditions([{"field": "short_signal", "op": "is", "value": "Buy"}])) is False
    assert is_intraday_eligible(parse_conditions([{"field": "sessions_since_earnings", "op": "<", "value": "3"}])) is False


def test_build_condition_frame_score_forward_fills_between_snapshots():
    ohlc, regime_by_date, score_history, _ = _replay_fixture()
    frame = build_condition_frame(ohlc.iloc[:201], regime_by_date, [], score_history)
    # Day 150 is between the two captured snapshots (day 0 and day 200)
    # -- must carry the day-0 value forward, not be null.
    assert frame["short_score"].iloc[150] == 50.0
    assert frame["short_score"].iloc[200] == 80.0
