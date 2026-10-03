from datetime import date

from services.challenge_service import (
    JOIN_CODE_ALPHABET,
    JOIN_CODE_LENGTH,
    compute_member_performance,
    generate_join_code,
)


def _snap(d: str, equity: float) -> dict:
    return {"as_of_date": date.fromisoformat(d), "equity": equity}


def test_generate_join_code_length_and_alphabet():
    code = generate_join_code()
    assert len(code) == JOIN_CODE_LENGTH
    assert all(c in JOIN_CODE_ALPHABET for c in code)


def test_generate_join_code_excludes_ambiguous_characters():
    for ambiguous in "0O1IL":
        assert ambiguous not in JOIN_CODE_ALPHABET


def test_compute_member_performance_too_little_data_returns_none_percentages():
    result = compute_member_performance(
        [_snap("2026-10-01", 100000.0)], date(2026, 10, 1), date(2026, 10, 31)
    )
    assert result["days_of_data"] == 1
    assert result["return_pct"] is None
    assert result["max_drawdown_pct"] is None
    assert result["annualized_volatility_pct"] is None


def test_compute_member_performance_no_snapshots_in_range():
    result = compute_member_performance([], date(2026, 10, 1), date(2026, 10, 31))
    assert result["days_of_data"] == 0
    assert result["return_pct"] is None


def test_compute_member_performance_computes_return_matching_simple_growth():
    snapshots = [_snap("2026-10-01", 100000.0), _snap("2026-10-02", 101000.0), _snap("2026-10-03", 102010.0)]
    result = compute_member_performance(snapshots, date(2026, 10, 1), date(2026, 10, 31))
    assert result["days_of_data"] == 3
    assert result["return_pct"] == 2.01  # 100000 -> 102010 is a flat +2.01%


def test_compute_member_performance_excludes_snapshots_outside_window():
    snapshots = [
        _snap("2026-09-28", 50000.0),  # before start_date -- should be ignored
        _snap("2026-10-01", 100000.0),
        _snap("2026-10-02", 110000.0),
        _snap("2026-11-05", 500000.0),  # after end_date -- should be ignored
    ]
    result = compute_member_performance(snapshots, date(2026, 10, 1), date(2026, 10, 31))
    assert result["days_of_data"] == 2
    assert result["return_pct"] == 10.0


def test_compute_member_performance_reflects_a_mid_period_dip_in_drawdown():
    snapshots = [
        _snap("2026-10-01", 100000.0),
        _snap("2026-10-02", 90000.0),   # -10% dip
        _snap("2026-10-03", 95000.0),   # partial recovery, still below peak
    ]
    result = compute_member_performance(snapshots, date(2026, 10, 1), date(2026, 10, 31))
    assert result["max_drawdown_pct"] == -10.0
    assert result["return_pct"] == -5.0


def test_compute_member_performance_unsorted_input_still_computed_correctly():
    snapshots = [
        _snap("2026-10-03", 102010.0),
        _snap("2026-10-01", 100000.0),
        _snap("2026-10-02", 101000.0),
    ]
    result = compute_member_performance(snapshots, date(2026, 10, 1), date(2026, 10, 31))
    assert result["return_pct"] == 2.01
