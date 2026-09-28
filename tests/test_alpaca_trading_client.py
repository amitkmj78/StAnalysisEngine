from unittest.mock import MagicMock, patch

import httpx
import pytest

from services.alpaca_trading_client import (
    AlpacaTradingError,
    get_account,
    get_order_by_client_order_id,
    submit_order,
)


def _response(status_code, json_body):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_body
    response.text = str(json_body)
    return response


@patch("services.alpaca_trading_client.httpx.get")
def test_get_account_sends_key_pair_headers_not_env_vars(mock_get):
    mock_get.return_value = _response(200, {"buying_power": "1000"})
    get_account("KEY123", "SECRET456")
    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {"APCA-API-KEY-ID": "KEY123", "APCA-API-SECRET-KEY": "SECRET456"}


@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_includes_client_order_id(mock_post):
    mock_post.return_value = _response(200, {"id": "broker-order-1", "status": "accepted"})
    submit_order(
        "KEY123", "SECRET456",
        client_order_id="abc-123", ticker="AAPL", side="buy", qty=1,
        order_type="market", time_in_force="day",
    )
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["client_order_id"] == "abc-123"
    assert kwargs["json"]["symbol"] == "AAPL"


@patch("services.alpaca_trading_client.httpx.get")
def test_definite_rejection_surfaces_as_alpaca_trading_error_with_message(mock_get):
    mock_get.return_value = _response(422, {"message": "insufficient buying power"})
    with pytest.raises(AlpacaTradingError) as exc_info:
        get_account("KEY123", "SECRET456")
    assert exc_info.value.status_code == 422
    assert "insufficient buying power" in exc_info.value.message


@patch("services.alpaca_trading_client.httpx.get")
def test_timeout_is_a_raw_exception_distinct_from_alpaca_trading_error(mock_get):
    mock_get.side_effect = httpx.ConnectTimeout("timed out")
    with pytest.raises(httpx.ConnectTimeout):
        get_account("KEY123", "SECRET456")


@patch("services.alpaca_trading_client.httpx.get")
def test_get_order_by_client_order_id_returns_none_on_404(mock_get):
    mock_get.return_value = _response(404, {"message": "not found"})
    result = get_order_by_client_order_id("KEY123", "SECRET456", "abc-123")
    assert result is None
