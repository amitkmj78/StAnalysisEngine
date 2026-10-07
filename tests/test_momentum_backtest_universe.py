from unittest.mock import patch

from services.momentum_backtest_service import _universe_tickers


def test_stock_all_current_only_resolves_via_live_sp500_fetch_not_empty_placeholder():
    """Regression test: _universe_tickers("Stock", "All", current_only=True)
    used to do a naive STOCK_UNIVERSES.get("All", []) lookup (back when
    this was the only path) -- but "All" and "US - S&P 500" are
    deliberately empty placeholders in that dict (see stock_finder_
    service.STOCK_UNIVERSES's own comment), meant to be resolved lazily
    via fetch_sp500_tickers(). The naive lookup silently returned zero
    tickers, which made every "Stock" backtest against "All" fail with
    "not enough historical data" -- 100% of the time, regardless of
    horizon_days/years/lookback_days, not a transient data issue."""
    with patch(
        "services.stock_finder_service.fetch_sp500_tickers",
        return_value=["AAA", "BBB", "CCC"],
    ):
        tickers, info = _universe_tickers("Stock", "All", current_only=True)
    assert len(tickers) > 0
    assert "AAA" in tickers
    assert info["basis"] == "current_members_biased"


def test_stock_sp500_current_only_resolves_via_live_fetch():
    with patch(
        "services.stock_finder_service.fetch_sp500_tickers",
        return_value=["AAA", "BBB", "CCC"],
    ):
        tickers, info = _universe_tickers("Stock", "US - S&P 500", current_only=True)
    assert tickers == ["AAA", "BBB", "CCC"]
    assert info["basis"] == "current_members_biased"


# --- NFR-1: default resolution is point-in-time, not today's membership ---

def test_stock_all_default_resolves_point_in_time_not_current():
    """Default (current_only=False) must NOT call today's live-membership
    fetch at all -- a ticker that has since left the S&P 500 should still
    be a candidate for a backtest window starting years ago."""
    with patch("services.momentum_backtest_service.members_on", return_value={"AAA", "BBB", "REMOVED"}), \
         patch("services.momentum_backtest_service.removed_after", return_value=["REMOVED"]), \
         patch("services.stock_finder_service.fetch_sp500_tickers") as fake_live_fetch:
        tickers, info = _universe_tickers("Stock", "All", years=3, current_only=False)

    fake_live_fetch.assert_not_called()
    assert set(tickers) == {"AAA", "BBB", "REMOVED"}
    assert info["basis"] == "point_in_time"
    assert info["left_index_in_window"] == ["REMOVED"]
    assert info["members_at_start"] == 3


def test_stock_sp500_default_falls_back_to_current_when_membership_history_missing():
    with patch("services.momentum_backtest_service.members_on", return_value=set()), \
         patch("services.stock_finder_service.fetch_sp500_tickers", return_value=["AAA"]):
        tickers, info = _universe_tickers("Stock", "US - S&P 500", years=3, current_only=False)
    assert tickers == ["AAA"]
    assert info["basis"] == "current_members_fallback"


def test_stock_named_sample_universe_still_works():
    tickers, info = _universe_tickers("Stock", "US - Mega Cap (SPY sample)")
    assert len(tickers) > 0
    assert info["basis"] == "current_members"


def test_fund_all_still_returns_every_fund():
    tickers, info = _universe_tickers("Fund", "All")
    assert len(tickers) > 0
    assert info["basis"] == "current_members"
