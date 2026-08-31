from unittest.mock import MagicMock, patch

import pytest

from services.alpaca_client import AlpacaSymbolNotFound, get_alpaca_latest_price


def _fake_env():
    return patch.dict(
        "os.environ", {"ALPACA_API_KEY_ID": "test-key", "ALPACA_API_SECRET_KEY": "test-secret"}
    )


def test_get_alpaca_latest_price_returns_none_when_not_configured():
    with patch.dict("os.environ", {}, clear=True):
        assert get_alpaca_latest_price("AAPL") is None


def test_get_alpaca_latest_price_parses_trade_price():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"trade": {"p": 234.567}}
    with _fake_env(), patch("services.alpaca_client.httpx.get", return_value=fake_response) as mock_get:
        price = get_alpaca_latest_price("AAPL")
    assert price == 234.57
    args, kwargs = mock_get.call_args
    assert "AAPL" in args[0]
    assert kwargs["headers"] == {"APCA-API-KEY-ID": "test-key", "APCA-API-SECRET-KEY": "test-secret"}


def test_get_alpaca_latest_price_returns_none_on_request_failure():
    with _fake_env(), patch("services.alpaca_client.httpx.get", side_effect=Exception("boom")):
        assert get_alpaca_latest_price("AAPL") is None


def test_get_alpaca_latest_price_returns_none_when_no_trade_in_response():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {}
    with _fake_env(), patch("services.alpaca_client.httpx.get", return_value=fake_response):
        assert get_alpaca_latest_price("ZZZZ") is None


def test_get_alpaca_latest_price_raises_symbol_not_found_on_404():
    """Mutual funds (FXAIX, CMIUX, ...) never trade on any exchange, so
    Alpaca's IEX feed always 404s for them — a confirmed, permanent
    answer, not a transient failure, so this must be distinguishable
    from the generic None-on-any-error case above."""
    fake_response = MagicMock()
    fake_response.status_code = 404
    with _fake_env(), patch("services.alpaca_client.httpx.get", return_value=fake_response):
        with pytest.raises(AlpacaSymbolNotFound):
            get_alpaca_latest_price("FXAIX")
