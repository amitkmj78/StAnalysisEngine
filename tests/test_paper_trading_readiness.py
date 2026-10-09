from datetime import datetime, timedelta, timezone

from services.paper_trading_readiness import MIN_DAYS_OPEN, MIN_TRADES, compute_readiness


def test_not_ready_with_no_paper_account_yet():
    result = compute_readiness(None, 0, False)
    assert result["ready"] is False
    assert result["days_open"] == 0
    assert f"{MIN_DAYS_OPEN} more day(s)" in result["missing"][0]
    assert any("paper trade" in m for m in result["missing"])
    assert "the risk quiz" in result["missing"]


def test_ready_once_all_three_conditions_are_met():
    old_enough = datetime.now(timezone.utc) - timedelta(days=MIN_DAYS_OPEN + 5)
    result = compute_readiness(old_enough, MIN_TRADES, True)
    assert result["ready"] is True
    assert result["missing"] == []


def test_partial_progress_lists_only_what_is_still_missing():
    almost_there = datetime.now(timezone.utc) - timedelta(days=MIN_DAYS_OPEN + 5)
    result = compute_readiness(almost_there, MIN_TRADES - 3, True)
    assert result["ready"] is False
    assert result["missing"] == ["3 more paper trade(s)"]


def test_naive_datetime_is_treated_as_utc_not_rejected():
    """alpaca_paper_accounts.created_at always comes back tz-aware from
    asyncpg, but this shouldn't blow up if a caller ever passes a naive
    datetime (e.g. in a test)."""
    naive_old_enough = (datetime.now(timezone.utc) - timedelta(days=MIN_DAYS_OPEN + 1)).replace(tzinfo=None)
    result = compute_readiness(naive_old_enough, MIN_TRADES, True)
    assert result["days_open"] >= MIN_DAYS_OPEN


def test_negative_trade_count_is_clamped_not_negative():
    result = compute_readiness(None, -5, False)
    assert result["trades_done"] == 0
