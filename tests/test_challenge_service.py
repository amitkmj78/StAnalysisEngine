import asyncio
from datetime import date
from unittest.mock import patch

from services.challenge_service import (
    DIVERSIFICATION_CONCENTRATION_LIMIT_PCT,
    rebase_to_100,
    score_for,
    JOIN_CODE_ALPHABET,
    JOIN_CODE_LENGTH,
    compute_member_diversification,
    compute_member_performance,
    generate_join_code,
)
from services.alpaca_trading_client import AlpacaTradingError


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


def test_risk_scores_blank_with_too_few_days():
    snaps = [_snap("2026-10-01", 100000.0), _snap("2026-10-02", 101000.0), _snap("2026-10-03", 99000.0)]
    result = compute_member_performance(snaps, date(2026, 10, 1), date(2026, 10, 31))
    assert result["sharpe"] is None and result["sortino"] is None and result["calmar"] is None
    assert result["return_pct"] is not None


def test_calmar_is_return_over_max_drawdown():
    snaps = [_snap(f"2026-10-{d:02d}", e) for d, e in
             [(1, 100000.0), (2, 110000.0), (3, 88000.0), (4, 120000.0), (5, 115000.0), (6, 121000.0)]]
    result = compute_member_performance(snaps, date(2026, 10, 1), date(2026, 10, 31))
    assert result["max_drawdown_pct"] == -20.0
    assert result["calmar"] == round(result["return_pct"] / 20.0, 2)


def test_score_for_return_and_excess_vs_spy():
    perf = {"return_pct": 5.0, "sharpe": 1.2, "sortino": 1.5, "calmar": 2.0}
    assert score_for("return", perf, 2.0) == 5.0
    assert score_for("excess_spy", perf, 2.0) == 3.0
    assert score_for("sharpe", perf, 2.0) == 1.2


def test_excess_return_is_unscored_without_spy_data():
    assert score_for("excess_spy", {"return_pct": 5.0}, None) is None
    assert score_for("excess_spy", {"return_pct": None}, 2.0) is None


def test_unknown_method_falls_back_to_raw_return():
    assert score_for("nonsense", {"return_pct": 4.0}, None) == 4.0


def test_rebase_starts_at_100_and_tracks_moves():
    series = rebase_to_100([(date(2026, 10, 2), 110.0), (date(2026, 10, 1), 100.0), (date(2026, 10, 3), 95.0)])
    assert [p["value"] for p in series] == [100.0, 110.0, 95.0]
    assert series[0]["date"] == "2026-10-01"


# BEG-6: beginner challenges' 'diversified' scoring method.

def test_diversified_scoring_displays_sortino():
    """The displayed/tie-breaking number is Sortino, a real metric --
    diversification affects rank via challenge_leaderboard.py's sort key,
    not by being blended into this number."""
    perf = {"return_pct": 5.0, "sharpe": 1.2, "sortino": 1.5, "calmar": 2.0}
    assert score_for("diversified", perf, 2.0) == 1.5


_ACCOUNT = {"id": 1, "api_key_id": "key", "api_secret_key_encrypted": "enc"}


def test_compute_member_diversification_flags_concentrated_account():
    positions = [{"symbol": "AAPL", "market_value": "8000"}, {"symbol": "MSFT", "market_value": "2000"}]
    with patch("services.challenge_service.decrypt_token", return_value="secret"), patch(
        "services.challenge_service.alpaca_trading_client.list_positions", return_value=positions
    ):
        result = asyncio.run(compute_member_diversification(_ACCOUNT))
    assert result["largest_position_pct"] == 80.0
    assert result["diversification_ok"] is False
    assert result["holdings_count"] == 2


def test_compute_member_diversification_ok_within_the_limit():
    # Evenly split across 5 positions -> 20% largest, under the 25% limit.
    positions = [{"symbol": f"T{i}", "market_value": "2000"} for i in range(5)]
    with patch("services.challenge_service.decrypt_token", return_value="secret"), patch(
        "services.challenge_service.alpaca_trading_client.list_positions", return_value=positions
    ):
        result = asyncio.run(compute_member_diversification(_ACCOUNT))
    assert result["largest_position_pct"] == 20.0
    assert result["diversification_ok"] is True
    assert result["largest_position_pct"] < DIVERSIFICATION_CONCENTRATION_LIMIT_PCT
    assert result["holdings_count"] == 5


def test_compute_member_diversification_exactly_at_the_limit_is_ok():
    # 4 equal positions -> the largest is exactly 25%, at the limit.
    positions = [{"symbol": f"T{i}", "market_value": "2500"} for i in range(4)]
    with patch("services.challenge_service.decrypt_token", return_value="secret"), patch(
        "services.challenge_service.alpaca_trading_client.list_positions", return_value=positions
    ):
        result = asyncio.run(compute_member_diversification(_ACCOUNT))
    assert result["largest_position_pct"] == DIVERSIFICATION_CONCENTRATION_LIMIT_PCT
    assert result["diversification_ok"] is True
    assert result["holdings_count"] == 4


def test_compute_member_diversification_defaults_to_ok_on_alpaca_failure():
    """"Can't tell" must never be treated as "concentrated" -- it would
    unfairly penalize a member over a transient API error, not their
    actual portfolio."""
    with patch("services.challenge_service.decrypt_token", return_value="secret"), patch(
        "services.challenge_service.alpaca_trading_client.list_positions",
        side_effect=AlpacaTradingError(500, "boom"),
    ):
        result = asyncio.run(compute_member_diversification(_ACCOUNT))
    assert result == {"largest_position_pct": None, "diversification_ok": True, "holdings_count": 0}


def test_compute_member_diversification_empty_account_is_ok():
    with patch("services.challenge_service.decrypt_token", return_value="secret"), patch(
        "services.challenge_service.alpaca_trading_client.list_positions", return_value=[]
    ):
        result = asyncio.run(compute_member_diversification(_ACCOUNT))
    assert result == {"largest_position_pct": None, "diversification_ok": True, "holdings_count": 0}


def test_rebase_handles_empty_and_zero_base():
    assert rebase_to_100([]) == []
    assert rebase_to_100([(date(2026, 10, 1), 0.0), (date(2026, 10, 2), 5.0)]) == []
