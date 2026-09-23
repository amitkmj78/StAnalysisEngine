"""
Fund Screener Priority 1 (FS-1 through FS-6): pure-function coverage for the
window resolution, per-window stats, and peer-group z-score scoring that
replaced the old fixed-1Y/3Y, whole-universe min-max normalization. No live
network — every test builds its own synthetic price series/DataFrame.
"""

import numpy as np
import pandas as pd
import pytest

from services.index_fund_service import (
    InvalidCustomWeights,
    _apply_peer_group_scores,
    _stats_for_window,
    _window_bounds,
    _zscore_series,
    normalize_custom_weights,
    rank_funds_overall,
)


def _prices(values, start="2020-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# _window_bounds
# ---------------------------------------------------------------------------


def test_window_bounds_fixed_window_ends_at_earliest_last_date():
    a = _prices([100] * 400, start="2023-01-01")
    b = _prices([100] * 300, start="2023-06-01")  # ends earlier than a
    start, end, error = _window_bounds("1y", {"A": a, "B": b})

    assert error is None
    assert end == min(a.index[-1], b.index[-1])
    assert start == end - pd.Timedelta(days=365)


def test_window_bounds_max_common_uses_latest_first_date():
    a = _prices([100] * 900, start="2020-01-01")  # long history, extends well past B's range
    b = _prices([100] * 100, start="2023-01-02")  # short, later-starting history
    start, end, error = _window_bounds("max_common", {"A": a, "B": b})

    assert error is None
    # The longest range every fund in the set actually has data for --
    # bounded by B's later start, not A's much earlier one.
    assert start == b.index[0]
    assert end == min(a.index[-1], b.index[-1])


def test_window_bounds_empty_price_series_returns_error():
    start, end, error = _window_bounds("5y", {"A": pd.Series(dtype=float)})
    assert start is None
    assert end is None
    assert error is not None


def test_window_bounds_no_overlap_returns_error():
    a = _prices([100] * 30, start="2020-01-01")
    b = _prices([100] * 30, start="2024-01-01")
    start, end, error = _window_bounds("max_common", {"A": a, "B": b})
    assert start is None
    assert end is None
    assert error is not None


def test_window_bounds_rejects_unknown_window():
    a = _prices([100] * 30)
    start, end, error = _window_bounds("2y", {"A": a})
    assert start is None
    assert end is None
    assert "window must be one of" in error


# ---------------------------------------------------------------------------
# _stats_for_window
# ---------------------------------------------------------------------------


def test_stats_for_window_known_flat_growth_path():
    # Doubles smoothly over exactly one simulated year (252 trading days) --
    # CAGR should land close to 100%, drawdown close to 0.
    days = 252
    values = np.linspace(100, 200, days)
    prices = _prices(values)
    start, end = prices.index[0], prices.index[-1]

    stats = _stats_for_window(prices, start, end)

    assert stats["cagr_window"] is not None
    assert 90 < stats["cagr_window"] < 110
    assert stats["max_drawdown_window"] is not None
    assert stats["max_drawdown_window"] < 1.0  # monotonic climb, near-zero drawdown
    assert stats["sharpe_window"] is not None
    assert stats["sharpe_window"] > 0


def test_stats_for_window_drawdown_matches_known_peak_to_trough():
    # 100 -> 150 -> 75: a clean 50% drawdown from the peak.
    values = [100, 110, 125, 140, 150, 130, 100, 90, 75, 80, 85, 90, 95, 100, 105,
              110, 115, 120, 118, 116, 114, 112, 110, 108, 106]
    prices = _prices(values)
    start, end = prices.index[0], prices.index[-1]

    stats = _stats_for_window(prices, start, end)
    assert stats["max_drawdown_window"] == pytest.approx(50.0, abs=0.01)


def test_stats_for_window_too_few_days_returns_all_none():
    prices = _prices([100, 101, 102, 103, 104])  # far fewer than 20 trading days
    start, end = prices.index[0], prices.index[-1]

    stats = _stats_for_window(prices, start, end)
    assert all(v is None for v in stats.values())


def test_stats_for_window_flat_prices_has_no_sharpe_but_no_crash():
    prices = _prices([100.0] * 40)
    start, end = prices.index[0], prices.index[-1]

    stats = _stats_for_window(prices, start, end)
    assert stats["cagr_window"] == pytest.approx(0.0, abs=0.01)
    assert stats["sharpe_window"] is None  # zero volatility -- undefined, not a fake number


# ---------------------------------------------------------------------------
# _zscore_series / _apply_peer_group_scores
# ---------------------------------------------------------------------------


def test_zscore_series_sign_flips_for_lower_is_better():
    values = pd.Series([1.0, 2.0, 3.0])
    higher_better = _zscore_series(values, lower_is_better=False)
    lower_better = _zscore_series(values, lower_is_better=True)

    assert higher_better.iloc[2] > higher_better.iloc[0]  # 3.0 scores best
    assert lower_better.iloc[0] > lower_better.iloc[2]  # 1.0 scores best when lower is better


def test_zscore_series_zero_std_returns_all_zero():
    values = pd.Series([5.0, 5.0, 5.0])
    z = _zscore_series(values, lower_is_better=False)
    assert (z == 0.0).all()


def test_apply_peer_group_scores_scores_within_category_not_across():
    """
    A fund with return_1y=10 among a Bond group averaging ~2 should score
    very highly, while the same absolute value of 10 among an Equity group
    averaging ~30 should score poorly -- proof that scoring is peer-group
    relative (FS-3), not against the whole result set.
    """
    df = pd.DataFrame(
        {
            "Ticker": ["BOND_HI", "BOND_A", "BOND_B", "EQ_HI", "EQ_A", "EQ_B"],
            "Category": ["Bond", "Bond", "Bond", "Equity", "Equity", "Equity"],
            "return_1y": [10.0, 1.0, 2.0, 10.0, 28.0, 32.0],
        }
    )
    scored = _apply_peer_group_scores(df, {"return_1y": 1.0})

    bond_hi_score = scored.loc[scored["Ticker"] == "BOND_HI", "Score"].iloc[0]
    eq_hi_score = scored.loc[scored["Ticker"] == "EQ_HI", "Score"].iloc[0]

    assert bond_hi_score > 0  # best in its own (low-return) peer group
    assert eq_hi_score < 0  # worst in its own (high-return) peer group


def test_apply_peer_group_scores_breakdown_matches_weighted_metrics():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "Category": ["US Large Blend", "US Large Blend"],
            "expense_ratio": [0.03, 0.10],
            "return_1y": [12.0, 8.0],
        }
    )
    scored = _apply_peer_group_scores(df, {"expense_ratio": 0.5, "return_1y": 0.5})

    breakdown_a = scored.loc[scored["Ticker"] == "A", "_breakdown"].iloc[0]
    assert "Cost" in breakdown_a
    assert "Return" in breakdown_a
    cost_metrics = [m["key"] for m in breakdown_a["Cost"]["metrics"]]
    assert cost_metrics == ["expense_ratio"]
    # Lower expense ratio (A) should contribute a positive Cost sub-score.
    assert breakdown_a["Cost"]["sub_score"] > 0


# ---------------------------------------------------------------------------
# Custom weight normalization (services.index_fund_service.normalize_custom_weights)
# ---------------------------------------------------------------------------


def test_normalize_custom_weights_normalizes_to_sum_one():
    weights = normalize_custom_weights({"return_1y": 3, "expense_ratio": 1})
    assert weights == pytest.approx({"return_1y": 0.75, "expense_ratio": 0.25})


def test_normalize_custom_weights_rejects_unknown_metric():
    with pytest.raises(InvalidCustomWeights):
        normalize_custom_weights({"not_a_real_metric": 1})


def test_normalize_custom_weights_rejects_all_zero_weights():
    with pytest.raises(InvalidCustomWeights):
        normalize_custom_weights({"return_1y": 0, "expense_ratio": 0})


def test_normalize_custom_weights_rejects_empty_input():
    with pytest.raises(InvalidCustomWeights):
        normalize_custom_weights({})


def test_normalize_custom_weights_rejects_negative_weight():
    with pytest.raises(InvalidCustomWeights):
        normalize_custom_weights({"return_1y": -1})


# ---------------------------------------------------------------------------
# rank_funds_overall -- regression test for a real, reported bug: callers
# that want "the best fund(s) overall" (get_top_fund, get_diverse_strategy_
# picks, etc.) were taking .head(n)/.iloc[0] directly off rank_index_funds's
# own Category-then-Score sorted output, so they silently returned whichever
# category sorted alphabetically first -- not the best-scoring funds. A fund
# with a peer-group Score of 0.0 (the correct result for a single-member
# category with no peers to compare against) was observed ranking #1 for
# every goal, ahead of a fund scoring 70+, purely because its category name
# came first alphabetically.
# ---------------------------------------------------------------------------


def test_rank_funds_overall_reorders_category_first_sort_by_score():
    # Mirrors rank_index_funds's own Category-then-Score sort: category "A"
    # sorts first alphabetically but only has a low-scoring fund, while
    # category "B" (sorted second) has the genuinely best-scoring fund.
    df = pd.DataFrame(
        {
            "Ticker": ["ALOW", "BHIGH", "BMID"],
            "Category": ["A", "B", "B"],
            "Score": [5.0, 90.0, 40.0],
            "1Y Return %": [1.0, 2.0, 3.0],
            "Assets ($B)": [1.0, 2.0, 3.0],
        }
    ).sort_values(["Category", "Score"], ascending=[True, False]).reset_index(drop=True)

    # Before the fix, a naive .iloc[0] on the Category-first-sorted df
    # would return ALOW (Score 5.0) simply because "A" < "B".
    assert df.iloc[0]["Ticker"] == "ALOW"

    overall = rank_funds_overall(df)
    assert overall.iloc[0]["Ticker"] == "BHIGH"
    assert list(overall["Ticker"]) == ["BHIGH", "BMID", "ALOW"]


def test_rank_funds_overall_empty_df_is_a_noop():
    df = pd.DataFrame()
    assert rank_funds_overall(df).empty
