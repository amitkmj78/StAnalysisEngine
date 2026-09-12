from unittest.mock import patch

from services.momentum_backtest_service import _universe_tickers


def test_stock_all_resolves_via_live_sp500_fetch_not_empty_placeholder():
    """Regression test: _universe_tickers("Stock", "All") used to do a
    naive STOCK_UNIVERSES.get("All", []) lookup -- but "All" and
    "US - S&P 500" are deliberately empty placeholders in that dict
    (see stock_finder_service.STOCK_UNIVERSES's own comment), meant to
    be resolved lazily via fetch_sp500_tickers(). The naive lookup
    silently returned zero tickers, which made every "Stock" backtest
    against "All" fail with "not enough historical data" -- 100% of the
    time, regardless of horizon_days/years/lookback_days, not a
    transient data issue."""
    with patch(
        "services.stock_finder_service.fetch_sp500_tickers",
        return_value=["AAA", "BBB", "CCC"],
    ):
        tickers = _universe_tickers("Stock", "All")
    assert len(tickers) > 0
    assert "AAA" in tickers


def test_stock_sp500_resolves_via_live_fetch():
    with patch(
        "services.stock_finder_service.fetch_sp500_tickers",
        return_value=["AAA", "BBB", "CCC"],
    ):
        tickers = _universe_tickers("Stock", "US - S&P 500")
    assert tickers == ["AAA", "BBB", "CCC"]


def test_stock_named_sample_universe_still_works():
    tickers = _universe_tickers("Stock", "US - Mega Cap (SPY sample)")
    assert len(tickers) > 0


def test_fund_all_still_returns_every_fund():
    tickers = _universe_tickers("Fund", "All")
    assert len(tickers) > 0
