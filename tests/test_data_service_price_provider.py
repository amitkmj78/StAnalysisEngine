from unittest.mock import patch

import pytest

from services.alpaca_client import AlpacaSymbolNotFound
from services.data_service import get_extended_hours_price, get_latest_price
from services.price_provider import set_price_provider


@pytest.fixture(autouse=True)
def _reset_provider():
    set_price_provider("yahoo")
    yield
    set_price_provider("yahoo")


def test_get_latest_price_uses_alpaca_when_selected():
    set_price_provider("alpaca")
    with patch("services.data_service.get_alpaca_latest_price", return_value=123.45) as mock_alpaca:
        price = get_latest_price("ALPACA_TEST_TICKER_1")
    assert price == 123.45
    mock_alpaca.assert_called_once_with("ALPACA_TEST_TICKER_1")


def test_get_latest_price_does_not_fall_back_to_yahoo_when_alpaca_has_no_quote():
    set_price_provider("alpaca")
    with patch("services.data_service.get_alpaca_latest_price", return_value=None) as mock_alpaca, patch(
        "services.data_service.fetch_with_backoff"
    ) as mock_yahoo:
        price = get_latest_price("ALPACA_TEST_TICKER_2")
    assert price is None
    mock_alpaca.assert_called_once()
    mock_yahoo.assert_not_called()


def test_get_latest_price_falls_back_to_yahoo_when_alpaca_confirms_symbol_not_found():
    """Mutual funds (FXAIX, ...) 404 on Alpaca permanently — that's the
    one case that should fall through to Yahoo, unlike a plain None."""
    set_price_provider("alpaca")
    with patch(
        "services.data_service.get_alpaca_latest_price", side_effect=AlpacaSymbolNotFound("FXAIX")
    ), patch("services.data_service.fetch_with_backoff", return_value=88.0):
        price = get_latest_price("ALPACA_TEST_TICKER_5")
    assert price == 88.0


def test_get_extended_hours_price_returns_none_when_alpaca_selected():
    set_price_provider("alpaca")
    assert get_extended_hours_price("ALPACA_TEST_TICKER_3") is None


def test_switching_provider_does_not_serve_stale_cached_result():
    """Regression test: get_latest_price is ttl_cache'd keyed only on
    ticker, not provider. Without clearing that cache on every switch (see
    price_provider.set_price_provider), a None cached from a
    not-configured Alpaca call would keep being served for the rest of
    the TTL window even after switching back to Yahoo — this is exactly
    what production hit during manual verification of this feature."""
    ticker = "ALPACA_TEST_TICKER_4"
    set_price_provider("alpaca")
    with patch("services.data_service.get_alpaca_latest_price", return_value=None):
        assert get_latest_price(ticker) is None

    set_price_provider("yahoo")
    with patch("services.data_service.fetch_with_backoff", return_value=42.0):
        assert get_latest_price(ticker) == 42.0
