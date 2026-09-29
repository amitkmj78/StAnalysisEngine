from datetime import date
from unittest.mock import patch

import pandas as pd

from services.signal_publication_service import (
    attach_excess_vs_spy,
    build_model_portfolio_series,
    build_spy_comparison_series,
    compute_avg_excess_vs_spy,
    compute_calibration,
    compute_outcome_metrics,
    compute_outcome_metrics_by_model_version,
    compute_outcome_metrics_by_signal,
    compute_spy_returns_for_dates,
    confidence_for_outcome,
    fetch_spy_close_series,
    worst_misses,
)
from services.signal_publication_service import _period_end_labels


def _row(ticker="AAPL", target_date=date(2026, 1, 5), rank=1, realized=2.0, benchmark=1.0, model_version_hash="v1"):
    return {
        "ticker": ticker,
        "target_date": target_date,
        "rank": rank,
        "realized_return_pct": realized,
        "benchmark_return_pct": benchmark,
        "beat_benchmark": realized > benchmark,
        "model_version_hash": model_version_hash,
    }


def test_compute_outcome_metrics_includes_avg_return_pct():
    rows = [_row(realized=4.0, benchmark=1.0), _row(ticker="MSFT", realized=-2.0, benchmark=1.0)]
    metrics = compute_outcome_metrics(rows)
    assert metrics["avg_return_pct"] == 1.0


def test_compute_outcome_metrics_empty_returns_none_avg():
    metrics = compute_outcome_metrics([])
    assert metrics["avg_return_pct"] is None


def test_compute_outcome_metrics_by_model_version_groups_separately():
    rows = [
        _row(model_version_hash="v1", realized=10.0, benchmark=0.0),
        _row(ticker="MSFT", model_version_hash="v1", realized=6.0, benchmark=0.0),
        _row(ticker="GOOG", model_version_hash="v2", realized=-4.0, benchmark=0.0),
    ]
    by_version = compute_outcome_metrics_by_model_version(rows)
    assert set(by_version) == {"v1", "v2"}
    assert by_version["v1"]["num_evaluated_picks"] == 2
    assert by_version["v2"]["num_evaluated_picks"] == 1
    assert by_version["v1"]["avg_return_pct"] == 8.0


def test_compute_outcome_metrics_by_model_version_missing_hash_grouped_unknown():
    rows = [_row(model_version_hash=None)]
    by_version = compute_outcome_metrics_by_model_version(rows)
    assert "unknown" in by_version


def test_compute_outcome_metrics_by_signal_single_buy_group():
    # published_signals has no signal-type column at all -- every row is
    # implicitly a Buy-ranked pick, so this is an honest single-group
    # label, not a fabricated split.
    rows = [
        _row(realized=10.0, benchmark=0.0),
        _row(ticker="MSFT", realized=6.0, benchmark=0.0),
    ]
    by_signal = compute_outcome_metrics_by_signal(rows)
    assert set(by_signal) == {"Buy"}
    assert by_signal["Buy"]["num_evaluated_picks"] == 2
    assert by_signal["Buy"]["avg_return_pct"] == 8.0


def test_compute_outcome_metrics_by_signal_empty_for_no_rows():
    assert compute_outcome_metrics_by_signal([]) == {}


def test_worst_misses_sorted_ascending_and_capped():
    rows = [_row(ticker=t, realized=r, benchmark=0.0) for t, r in [("A", 5.0), ("B", -20.0), ("C", -1.0), ("D", -30.0)]]
    misses = worst_misses(rows, top_n=2)
    assert [m["ticker"] for m in misses] == ["D", "B"]


def _spy_series(prices: list[float], start=date(2026, 1, 1)) -> pd.Series:
    dates = pd.date_range(start=start, periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


def test_compute_spy_returns_for_dates_matches_known_return():
    # 20 trading days, flat at 100 except day 10 jumps to 110.
    prices = [100.0] * 20
    prices[10] = 110.0
    spy = _spy_series(prices)
    target_date = spy.index[0].date()
    result = compute_spy_returns_for_dates(spy, [target_date], horizon_days=10)
    # entry = index 0 (100.0), exit = index 10 (110.0) -> +10%
    assert result[target_date] == 10.0


def test_compute_spy_returns_for_dates_none_when_window_not_elapsed():
    prices = [100.0] * 5
    spy = _spy_series(prices)
    target_date = spy.index[0].date()
    result = compute_spy_returns_for_dates(spy, [target_date], horizon_days=10)
    assert result[target_date] is None


def test_attach_and_average_excess_vs_spy():
    rows = [_row(target_date=date(2026, 1, 5), realized=5.0), _row(ticker="MSFT", target_date=date(2026, 1, 6), realized=1.0)]
    spy_by_date = {date(2026, 1, 5): 2.0, date(2026, 1, 6): 3.0}
    with_excess = attach_excess_vs_spy(rows, spy_by_date)
    assert with_excess[0]["excess_vs_spy_pct"] == 3.0
    assert with_excess[1]["excess_vs_spy_pct"] == -2.0
    assert compute_avg_excess_vs_spy(with_excess) == 0.5


def test_attach_excess_vs_spy_none_when_spy_unresolved():
    rows = [_row(target_date=date(2026, 1, 5), realized=5.0)]
    with_excess = attach_excess_vs_spy(rows, {})
    assert with_excess[0]["excess_vs_spy_pct"] is None
    assert compute_avg_excess_vs_spy(with_excess) is None


def test_confidence_for_outcome_never_uses_publication_after_target_date():
    target_date = date(2026, 1, 20)
    all_dates = [
        date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 20),
        date(2026, 1, 21), date(2026, 1, 22), date(2026, 1, 23),
    ]
    # Ranked every day up to and including target_date (stable, no flips),
    # then flips wildly right after -- a bug that looked ahead would see
    # those later flips and downgrade confidence; this must not happen.
    ranked_dates = {date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 20), date(2026, 1, 22)}
    confidence = confidence_for_outcome(ranked_dates, all_dates, target_date, lookback_days=30)
    assert confidence["label"] == "high"


def test_confidence_for_outcome_reflects_instability_before_target_date():
    target_date = date(2026, 1, 20)
    all_dates = [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 10), date(2026, 1, 15), date(2026, 1, 20)]
    # In the top-N, then out, then in, then out, then in -- 4 flips across
    # 5 points, well past the unstable threshold.
    ranked_dates = {date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 20)}
    confidence = confidence_for_outcome(ranked_dates, all_dates, target_date, lookback_days=30)
    assert confidence["label"] == "low"


def test_confidence_for_outcome_none_with_insufficient_window_history():
    target_date = date(2026, 1, 20)
    confidence = confidence_for_outcome({target_date}, [target_date], target_date)
    assert confidence == {"label": "unknown", "score": None}


def test_confidence_for_outcome_unranked_days_count_too():
    # A ticker that's never in the top-N over the window is just as
    # "stable" (0 flips, all "Unranked") as one that's always in it --
    # confidence isn't only measuring presence, it's measuring
    # consistency either way.
    target_date = date(2026, 1, 20)
    all_dates = [date(2026, 1, 1), date(2026, 1, 10), date(2026, 1, 20)]
    confidence = confidence_for_outcome(set(), all_dates, target_date, lookback_days=30)
    assert confidence["label"] == "high"


def test_compute_calibration_buckets_hit_rate_and_sample_size():
    rows = [
        {"beat_benchmark": True, "confidence_score": 95},
        {"beat_benchmark": True, "confidence_score": 92},
        {"beat_benchmark": False, "confidence_score": 91},
        {"beat_benchmark": True, "confidence_score": 55},
    ]
    buckets = compute_calibration(rows)
    top_bucket = next(b for b in buckets if b["bucket_label"] == "90-100%")
    assert top_bucket["sample_size"] == 3
    assert round(top_bucket["hit_rate_pct"], 1) == round(2 / 3 * 100, 1)
    low_bucket = next(b for b in buckets if b["bucket_label"] == "50-60%")
    assert low_bucket["sample_size"] == 1
    assert low_bucket["hit_rate_pct"] == 100.0


def test_compute_calibration_empty_bucket_reports_zero_sample():
    buckets = compute_calibration([{"beat_benchmark": True, "confidence_score": 95}])
    empty_bucket = next(b for b in buckets if b["bucket_label"] == "60-70%")
    assert empty_bucket == {"bucket_label": "60-70%", "hit_rate_pct": None, "sample_size": 0}


def test_compute_calibration_ignores_rows_with_no_confidence():
    buckets = compute_calibration([{"beat_benchmark": True, "confidence_score": None}])
    assert all(b["sample_size"] == 0 for b in buckets)


def test_build_model_portfolio_series_starts_at_10000_and_compounds():
    rows = [
        _row(target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
        _row(ticker="MSFT", target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
    ]
    # cost_bps_one_way=0 isolates the raw compounding math from the cost
    # deduction, which has its own dedicated test below.
    series = build_model_portfolio_series(rows, horizon_days=1, cost_bps_one_way=0.0)
    assert series[0] == ["2026-01-01", 10000.0]
    # Only one selected date -> its result is labeled with an estimated
    # period-end date, never the SAME date as the starting point (that
    # duplicate-x-value bug broke the chart's date axis in production).
    assert series[1][0] != "2026-01-01"
    assert series[1][1] == 11000.0


def test_build_model_portfolio_series_applies_trading_cost_each_period():
    rows = [
        _row(target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
        _row(ticker="MSFT", target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
    ]
    # 100 bps one-way -> 200 bps (2%) round-trip haircut on the period's
    # compounded value: 10000 * 1.10 * (1 - 0.02) = 10780.0.
    series = build_model_portfolio_series(rows, horizon_days=1, cost_bps_one_way=100.0)
    assert series[1][1] == 10780.0


def test_build_model_portfolio_series_default_cost_is_nonzero():
    # Regression: TRK-6 requires "after assumed trading costs" -- the
    # default must actually deduct something, not silently stay at the
    # old zero-cost behavior.
    rows = [
        _row(target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
        _row(ticker="MSFT", target_date=date(2026, 1, 1), realized=10.0, benchmark=0.0),
    ]
    series = build_model_portfolio_series(rows, horizon_days=1)
    assert series[1][1] < 11000.0


def test_build_model_portfolio_series_selects_non_overlapping_dates():
    # 5 published dates, horizon_days=2 -> only dates[0], dates[2], dates[4] used.
    rows = []
    for i, d in enumerate([date(2026, 1, 1 + i) for i in range(5)]):
        rows.append(_row(target_date=d, realized=float(i), benchmark=0.0))
    series = build_model_portfolio_series(rows, horizon_days=2)
    # starting point + 3 selected dates = 4 points
    assert len(series) == 4


def test_build_model_portfolio_series_empty_input():
    assert build_model_portfolio_series([], horizon_days=10) == []


def test_build_spy_comparison_series_matches_model_portfolio_dates():
    rows = [_row(target_date=date(2026, 1, 1 + i), realized=1.0, benchmark=0.0) for i in range(5)]
    spy_by_date = {date(2026, 1, 1): 5.0, date(2026, 1, 3): -2.0, date(2026, 1, 5): 1.0}
    model_series = build_model_portfolio_series(rows, horizon_days=2)
    spy_series = build_spy_comparison_series(rows, spy_by_date, horizon_days=2)
    model_dates = [p[0] for p in model_series]
    spy_dates = [p[0] for p in spy_series]
    assert model_dates == spy_dates
    # Every point has a distinct date -- no two points share an x-value.
    assert len(set(spy_dates)) == len(spy_dates)
    assert spy_series[0] == ["2026-01-01", 10000.0]
    # First selected date is 2026-01-01; its period-end label is the NEXT
    # selected date (2026-01-03), not the same date it started on.
    assert spy_series[1] == ["2026-01-03", 10500.0]


def test_build_spy_comparison_series_holds_value_when_spy_unresolved():
    rows = [_row(target_date=date(2026, 1, 1), realized=1.0, benchmark=0.0)]
    series = build_spy_comparison_series(rows, {}, horizon_days=1)
    assert series[0] == ["2026-01-01", 10000.0]
    assert series[1][0] != "2026-01-01"
    assert series[1][1] == 10000.0


def test_build_spy_comparison_series_empty_input():
    assert build_spy_comparison_series([], {}, horizon_days=10) == []


def test_period_end_labels_uses_next_selected_date():
    selected = [date(2026, 1, 1), date(2026, 1, 5), date(2026, 1, 9)]
    labels = _period_end_labels(selected, horizon_days=2)
    assert labels[0] == date(2026, 1, 5)
    assert labels[1] == date(2026, 1, 9)


def test_period_end_labels_estimates_final_period():
    labels = _period_end_labels([date(2026, 1, 1)], horizon_days=10)
    assert labels[0] > date(2026, 1, 1)


def test_build_model_portfolio_series_never_repeats_a_date():
    rows = [_row(target_date=date(2026, 1, 1 + i), realized=1.0, benchmark=0.0) for i in range(6)]
    series = build_model_portfolio_series(rows, horizon_days=1)
    dates = [p[0] for p in series]
    assert len(set(dates)) == len(dates)


def test_fetch_spy_close_series_strips_timezone_to_avoid_naive_vs_aware_crash():
    # Regression test: yfinance returns a tz-aware DatetimeIndex; comparing
    # it against a plain datetime.date via pd.Timestamp() in
    # compute_spy_returns_for_dates raised TypeError in production on the
    # very first live call to /signals/track-record. fetch_spy_close_series
    # must normalize to tz-naive before compute_spy_returns_for_dates ever
    # sees it.
    tz_aware_index = pd.date_range("2026-01-01", periods=5, freq="B", tz="America/New_York")
    fake_hist = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=tz_aware_index)
    with patch("services.signal_publication_service.get_cached_history", return_value=fake_hist):
        close = fetch_spy_close_series()
    assert close.index.tz is None
    # Must not raise -- this is exactly what crashed before the fix.
    result = compute_spy_returns_for_dates(close, [date(2026, 1, 1)], horizon_days=2)
    assert result[date(2026, 1, 1)] is not None
