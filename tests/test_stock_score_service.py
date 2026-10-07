from datetime import date

from services.stock_score_service import (
    LONG_TERM_WEIGHTS,
    compute_factor_contributions,
    compute_long_score,
    compute_short_score,
    flag_12week_trend,
    flip_count_from_signal_history,
    percentile_rank,
    score_to_signal,
    sector_percentile,
    sector_rank,
    select_top_and_bottom_factors,
    weekly_change_explanation,
    weights_version,
)


# --- NFR-4: reproducibility -- a stored row must be tied to the weights
# that produced it, same config_version() pattern services/agent/
# config.py already uses for the trading agent's own limits.

def test_weights_version_is_stable_and_changes_with_the_weights(monkeypatch):
    before = weights_version()
    assert before == weights_version()  # stable across repeated calls, same inputs

    original_value_weight = LONG_TERM_WEIGHTS["value"]
    monkeypatch.setitem(LONG_TERM_WEIGHTS, "value", original_value_weight + 0.01)
    assert weights_version() != before


def test_percentile_rank_higher_is_better():
    result = percentile_rank({"A": 10.0, "B": 20.0, "C": 30.0})
    assert result["C"] > result["B"] > result["A"]


def test_percentile_rank_lower_is_better_inverts():
    result = percentile_rank({"A": 10.0, "B": 20.0, "C": 30.0}, lower_is_better=True)
    assert result["A"] > result["B"] > result["C"]


def test_percentile_rank_none_passes_through():
    result = percentile_rank({"A": 10.0, "B": None})
    assert result["B"] is None
    assert result["A"] is not None


def test_percentile_rank_all_none_returns_all_none():
    result = percentile_rank({"A": None, "B": None})
    assert result == {"A": None, "B": None}


def test_compute_short_score_weighted_average():
    momentum = {"AAPL": 80.0, "MSFT": 20.0}
    reversal = {"AAPL": 60.0, "MSFT": 40.0}
    earnings_surprise = {"AAPL": 90.0, "MSFT": 10.0}
    earnings_revisions = {"AAPL": 50.0, "MSFT": 50.0}
    scores = compute_short_score(momentum, reversal, earnings_surprise, earnings_revisions)
    # 0.35*80 + 0.25*60 + 0.20*90 + 0.20*50 = 28 + 15 + 18 + 10 = 71
    assert scores["AAPL"] == 71.0
    # 0.35*20 + 0.25*40 + 0.20*10 + 0.20*50 = 7 + 10 + 2 + 10 = 29
    assert scores["MSFT"] == 29.0


def test_compute_short_score_renormalizes_when_a_factor_is_missing():
    momentum = {"AAPL": 80.0}
    reversal = {"AAPL": None}
    earnings_surprise = {"AAPL": None}
    earnings_revisions = {"AAPL": None}
    scores = compute_short_score(momentum, reversal, earnings_surprise, earnings_revisions)
    # Only momentum available -> renormalized weight is 100% momentum.
    assert scores["AAPL"] == 80.0


def test_compute_short_score_renormalizes_with_two_factors_missing():
    momentum = {"AAPL": 80.0}
    reversal = {"AAPL": None}
    earnings_surprise = {"AAPL": 60.0}
    earnings_revisions = {"AAPL": None}
    scores = compute_short_score(momentum, reversal, earnings_surprise, earnings_revisions)
    # (0.35*80 + 0.20*60) / (0.35+0.20) = (28+12)/0.55 = 72.7272... -> 72.73
    assert scores["AAPL"] == 72.73


def test_compute_long_score_weighted_average():
    value = {"T": 100.0}
    growth = {"T": 0.0}
    low_vol = {"T": 50.0}
    quality = {"T": 80.0}
    scores = compute_long_score(value, growth, low_vol, quality)
    # 0.30*100 + 0.25*0 + 0.20*50 + 0.25*80 = 30 + 0 + 10 + 20 = 60
    assert scores["T"] == 60.0


def test_score_to_signal_boundaries():
    assert score_to_signal(70.0) == "Buy"
    assert score_to_signal(69.99) == "Hold"
    assert score_to_signal(30.0) == "Trim"
    assert score_to_signal(30.01) == "Hold"
    assert score_to_signal(None) == "Hold"


def test_sector_percentile_scoped_per_sector():
    scores = {"A": 90.0, "B": 10.0, "C": 90.0, "D": 10.0}
    sector_of = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    result = sector_percentile(scores, sector_of)
    assert result["A"] > result["B"]
    assert result["C"] > result["D"]
    # A and C are both the top of their own sector -> same percentile.
    assert result["A"] == result["C"]


def test_sector_rank_orders_within_sector():
    scores = {"A": 90.0, "B": 50.0, "C": 10.0}
    sector_of = {"A": "Tech", "B": "Tech", "C": "Tech"}
    result = sector_rank(scores, sector_of)
    assert result["A"] == {"rank": 1, "of": 3}
    assert result["B"] == {"rank": 2, "of": 3}
    assert result["C"] == {"rank": 3, "of": 3}


def test_sector_rank_ties_share_the_same_rank():
    scores = {"A": 90.0, "B": 90.0, "C": 10.0}
    sector_of = {"A": "Tech", "B": "Tech", "C": "Tech"}
    result = sector_rank(scores, sector_of)
    assert result["A"] == {"rank": 1, "of": 3}
    assert result["B"] == {"rank": 1, "of": 3}
    assert result["C"] == {"rank": 3, "of": 3}


def test_sector_rank_scoped_per_sector_independently():
    scores = {"A": 90.0, "B": 10.0, "C": 90.0, "D": 10.0}
    sector_of = {"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"}
    result = sector_rank(scores, sector_of)
    # A and C are both the top of their own (2-member) sector.
    assert result["A"] == {"rank": 1, "of": 2}
    assert result["C"] == {"rank": 1, "of": 2}
    assert result["B"] == {"rank": 2, "of": 2}


def test_sector_rank_none_for_missing_score_or_sector():
    scores = {"A": 90.0, "B": None}
    sector_of = {"A": None, "B": "Tech"}
    result = sector_rank(scores, sector_of)
    assert result["A"] is None
    assert result["B"] is None


def test_compute_factor_contributions_sums_exactly_to_score_minus_50():
    percentiles = {"momentum": 80.0, "reversal": 60.0}
    raw = {"momentum": 5.0, "reversal": 55.0}
    weights = {"momentum": 0.6, "reversal": 0.4}
    contributions = compute_factor_contributions(raw, percentiles, weights)
    total = sum(c["contribution"] for c in contributions)
    # 0.6*(80-50) + 0.4*(60-50) = 18 + 4 = 22 = score(72) - 50
    assert round(total, 2) == 22.0
    assert contributions[0]["factor"] == "momentum"  # larger contribution first


def test_compute_factor_contributions_skips_missing_percentile():
    percentiles = {"momentum": 80.0, "reversal": None}
    raw = {"momentum": 5.0, "reversal": None}
    weights = {"momentum": 0.6, "reversal": 0.4}
    contributions = compute_factor_contributions(raw, percentiles, weights)
    assert len(contributions) == 1
    assert contributions[0]["factor"] == "momentum"


def test_compute_factor_contributions_exactly_additive_when_a_factor_is_missing():
    # Regression: contributions used to be only approximately additive to
    # score-50 for a ticker missing a factor, since the original
    # (non-renormalized) weight was used here while the score itself
    # (_weighted_composite) renormalizes over only the factors actually
    # available for that ticker. Now both use the identical
    # renormalization, so they stay in exact lockstep.
    percentiles = {"momentum": 80.0, "reversal": None}
    raw = {"momentum": 5.0, "reversal": None}
    weights = {"momentum": 0.6, "reversal": 0.4}
    contributions = compute_factor_contributions(raw, percentiles, weights)
    total = sum(c["contribution"] for c in contributions)
    # Only momentum available -> renormalized to 100% momentum weight,
    # so the score itself is just 80.0 (see
    # test_compute_short_score_renormalizes_when_a_factor_is_missing).
    # score - 50 = 30.0.
    assert total == 30.0


def test_compute_factor_contributions_exactly_additive_with_three_of_four_factors():
    percentiles = {"momentum": 80.0, "reversal": 60.0, "earnings_surprise": None, "earnings_revisions": 40.0}
    raw = {"momentum": 5.0, "reversal": 55.0, "earnings_surprise": None, "earnings_revisions": -1.0}
    weights = {"momentum": 0.35, "reversal": 0.25, "earnings_surprise": 0.20, "earnings_revisions": 0.20}
    contributions = compute_factor_contributions(raw, percentiles, weights)
    total = sum(c["contribution"] for c in contributions)
    # available_weight = 0.35+0.25+0.20 = 0.80
    # score = (0.35*80 + 0.25*60 + 0.20*40) / 0.80 = (28+15+8)/0.80 = 63.75
    available_weight = 0.35 + 0.25 + 0.20
    score = (0.35 * 80.0 + 0.25 * 60.0 + 0.20 * 40.0) / available_weight
    # Each contribution rounds to 2dp independently before summing, so the
    # total can be off from the analytically exact (score - 50) by a few
    # hundredths -- that's normal rounding drift, not the renormalization
    # bug this test exists to catch, so this checks "matches within cents"
    # rather than bit-exact equality.
    assert abs(total - (score - 50.0)) < 0.02


def test_select_top_and_bottom_factors():
    contributions = [
        {"factor": "momentum", "contribution": 10.0},
        {"factor": "reversal", "contribution": 5.0},
        {"factor": "value", "contribution": -8.0},
        {"factor": "growth", "contribution": -3.0},
        {"factor": "low_vol", "contribution": -1.0},
    ]
    result = select_top_and_bottom_factors(contributions, top_n=3, bottom_n=2)
    assert [d["factor"] for d in result["drivers"]] == ["momentum", "reversal"]
    assert [d["factor"] for d in result["drags"]] == ["value", "growth"]


def test_flip_count_from_signal_history_counts_transitions():
    history = [
        (date(2026, 1, 1), "Buy"),
        (date(2026, 1, 2), "Buy"),
        (date(2026, 1, 3), "Hold"),
        (date(2026, 1, 4), "Buy"),
    ]
    result = flip_count_from_signal_history(history)
    assert result["flip_count"] == 2
    assert result["days_captured"] == 4
    assert result["current_streak_days"] == 1
    assert result["unstable"] is False


def test_flip_count_from_signal_history_none_with_fewer_than_two_days():
    assert flip_count_from_signal_history([(date(2026, 1, 1), "Buy")]) is None


def test_flag_12week_trend_flags_at_exactly_15pts():
    history = [(date(2026, 1, 5), 50.0), (date(2026, 1, 12), 65.0)]
    result = flag_12week_trend(history, threshold_pts=15.0)
    assert result["flagged"] is True
    assert result["change_pts"] == 15.0


def test_flag_12week_trend_not_flagged_under_threshold():
    history = [(date(2026, 1, 5), 50.0), (date(2026, 1, 12), 64.0)]
    result = flag_12week_trend(history, threshold_pts=15.0)
    assert result["flagged"] is False


def test_flag_12week_trend_empty_history():
    result = flag_12week_trend([])
    assert result == {"weekly_series": [], "flagged": False, "change_pts": None}


def test_weekly_change_explanation_picks_biggest_mover():
    today = {"momentum": {"contribution": 10.0}, "value": {"contribution": 2.0}}
    week_ago = {"momentum": {"contribution": 1.0}, "value": {"contribution": 1.5}}
    result = weekly_change_explanation(today, week_ago)
    assert result["factor"] == "momentum"
    assert result["delta_contribution"] == 9.0


def test_weekly_change_explanation_none_without_week_ago_snapshot():
    assert weekly_change_explanation({"momentum": {"contribution": 10.0}}, None) is None
