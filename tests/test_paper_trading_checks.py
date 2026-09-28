from services.paper_trading_checks import (
    check_buying_power,
    check_daily_order_count,
    check_order_value_limit,
    check_portfolio_pct_limit,
    check_price_collar,
    check_restricted_symbol,
    estimate_order_value,
)


def test_estimate_order_value_uses_limit_price_for_limit_orders():
    assert estimate_order_value(10, "limit", 100.0, 90.0) == 1000.0


def test_estimate_order_value_uses_last_price_for_market_orders():
    assert estimate_order_value(10, "market", None, 90.0) == 900.0


def test_estimate_order_value_none_when_no_price_available():
    assert estimate_order_value(10, "market", None, None) is None


def test_buying_power_check_passes_when_estimate_under_available_cash():
    result = check_buying_power("buy", 10, 900.0, {"buying_power": "1000"}, shares_held=0)
    assert result.passed


def test_buying_power_check_fails_when_estimate_exceeds_buying_power():
    result = check_buying_power("buy", 10, 1500.0, {"buying_power": "1000"}, shares_held=0)
    assert not result.passed
    assert result.code == "insufficient_buying_power"


def test_sell_check_fails_when_qty_exceeds_shares_held():
    result = check_buying_power("sell", 10, 900.0, {"buying_power": "1000"}, shares_held=5)
    assert not result.passed
    assert result.code == "insufficient_shares"


def test_sell_check_passes_when_qty_within_shares_held():
    result = check_buying_power("sell", 5, 900.0, {"buying_power": "0"}, shares_held=10)
    assert result.passed


def test_order_value_limit_fails_when_estimate_exceeds_max():
    result = check_order_value_limit(11_000.0, max_order_value=10_000.0)
    assert not result.passed


def test_order_value_limit_passes_when_estimate_unknown():
    result = check_order_value_limit(None, max_order_value=10_000.0)
    assert result.passed


def test_portfolio_pct_limit_fails_when_estimate_exceeds_pct_of_equity():
    result = check_portfolio_pct_limit(3000.0, portfolio_equity=10_000.0, max_pct=25.0)
    assert not result.passed


def test_portfolio_pct_limit_passes_within_bound():
    result = check_portfolio_pct_limit(2000.0, portfolio_equity=10_000.0, max_pct=25.0)
    assert result.passed


def test_daily_order_count_fails_when_at_max():
    result = check_daily_order_count(20, max_orders_per_day=20)
    assert not result.passed


def test_daily_order_count_passes_below_max():
    result = check_daily_order_count(5, max_orders_per_day=20)
    assert result.passed


def test_price_collar_warns_when_limit_price_far_from_last_trade():
    result = check_price_collar("limit", 80.0, last_price=100.0, collar_pct=10.0)
    assert not result.passed
    assert result.code == "price_collar"


def test_price_collar_passes_when_limit_price_within_band():
    result = check_price_collar("limit", 98.0, last_price=100.0, collar_pct=10.0)
    assert result.passed


def test_price_collar_skipped_for_market_orders():
    result = check_price_collar("market", None, last_price=100.0, collar_pct=10.0)
    assert result.passed


def test_restricted_symbol_fails_when_ticker_in_blocklist():
    result = check_restricted_symbol("gme", {"GME", "AMC"})
    assert not result.passed


def test_restricted_symbol_passes_when_ticker_not_in_blocklist():
    result = check_restricted_symbol("AAPL", {"GME", "AMC"})
    assert result.passed
