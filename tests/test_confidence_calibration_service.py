from datetime import date, timedelta

from services.confidence_calibration_service import (
    AGREEMENT_THRESHOLD_POINTS,
    fit_calibration_table,
    validate_calibration_out_of_sample,
)


def _row(score, hit, target_date=date(2026, 1, 1)):
    return {"confidence_score": score, "beat_benchmark": hit, "target_date": target_date}


def test_fit_calibration_table_computes_hit_rate_per_score():
    rows = [_row(75, True)] * 20 + [_row(75, False)] * 10 + [_row(50, True)] * 30
    table = fit_calibration_table(rows, min_samples=30)
    assert table[75]["hit_rate_pct"] == 66.7
    assert table[75]["n"] == 30
    assert table[50]["hit_rate_pct"] == 100.0


def test_fit_calibration_table_omits_scores_below_min_samples():
    rows = [_row(100, True)] * 5
    table = fit_calibration_table(rows, min_samples=30)
    assert table == {}


def test_fit_calibration_table_ignores_rows_with_no_confidence_score():
    rows = [{"confidence_score": None, "beat_benchmark": True, "target_date": date(2026, 1, 1)}] * 40
    table = fit_calibration_table(rows, min_samples=30)
    assert table == {}


def test_validate_calibration_out_of_sample_splits_chronologically_not_randomly():
    # 100 rows, scores assigned by DATE order so a chronological split
    # (not a random one) produces a clean fit/holdout boundary we can
    # reason about directly.
    base = date(2026, 1, 1)
    rows = [_row(75, True, target_date=base + timedelta(days=i)) for i in range(60)]
    rows += [_row(75, False, target_date=base + timedelta(days=60 + i)) for i in range(40)]
    report = validate_calibration_out_of_sample(rows, fit_fraction=0.7, min_samples=30)
    # fit = first 70 rows (60 hits + first 10 misses = 85.7%), holdout =
    # last 30 rows (all misses = 0%) -- a split by DATE, not by the
    # original hit/miss row order, which is exactly the point.
    assert report["fit_set_size"] == 70
    assert report["holdout_set_size"] == 30
    assert report["buckets"][75]["fit_hit_rate_pct"] == 85.7
    assert report["buckets"][75]["holdout_hit_rate_pct"] == 0.0
    assert report["buckets"][75]["agrees_within_5_points"] is False


def test_validate_calibration_out_of_sample_agrees_when_within_5_points():
    base = date(2026, 1, 1)
    rows = [_row(50, i % 10 != 0, target_date=base + timedelta(days=i)) for i in range(70)]  # 90% hit rate
    rows += [_row(50, i % 10 != 0, target_date=base + timedelta(days=100 + i)) for i in range(30)]  # 90% hit rate
    report = validate_calibration_out_of_sample(rows, fit_fraction=0.7, min_samples=30)
    bucket = report["buckets"][50]
    assert abs(bucket["fit_hit_rate_pct"] - bucket["holdout_hit_rate_pct"]) <= AGREEMENT_THRESHOLD_POINTS
    assert bucket["agrees_within_5_points"] is True


def test_validate_calibration_out_of_sample_insufficient_data_reports_none_not_a_guess():
    base = date(2026, 1, 1)
    rows = [_row(75, True, target_date=base + timedelta(days=i)) for i in range(5)]
    report = validate_calibration_out_of_sample(rows, fit_fraction=0.7, min_samples=30)
    # Too few rows on either side of the split for a reliable number --
    # the bucket is still reported (so a caller can see "5 isn't enough"
    # rather than it silently vanishing), but with None/0, never a guess.
    bucket = report["buckets"][75]
    assert bucket["fit_hit_rate_pct"] is None
    assert bucket["holdout_hit_rate_pct"] is None
    assert bucket["agrees_within_5_points"] is None


def test_validate_calibration_out_of_sample_empty_input():
    report = validate_calibration_out_of_sample([])
    assert report["fit_set_size"] == 0
    assert report["holdout_set_size"] == 0
    assert report["buckets"] == {}
