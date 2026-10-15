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


# ---------------------------------------------------------------- BRK-2 order types

@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_stop_order_payload(mock_post):
    mock_post.return_value = _response(200, {"id": "o1"})
    submit_order(
        "K", "S", client_order_id="c1", ticker="AAPL", side="sell", qty=10,
        order_type="stop", time_in_force="day", stop_price=95.0,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["type"] == "stop"
    assert payload["stop_price"] == "95.0"
    assert "limit_price" not in payload


@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_stop_limit_order_payload(mock_post):
    mock_post.return_value = _response(200, {"id": "o1"})
    submit_order(
        "K", "S", client_order_id="c1", ticker="AAPL", side="sell", qty=10,
        order_type="stop_limit", time_in_force="day", stop_price=95.0, limit_price=94.0,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["type"] == "stop_limit"
    assert payload["stop_price"] == "95.0"
    assert payload["limit_price"] == "94.0"


@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_trailing_stop_order_payload(mock_post):
    mock_post.return_value = _response(200, {"id": "o1"})
    submit_order(
        "K", "S", client_order_id="c1", ticker="AAPL", side="sell", qty=10,
        order_type="trailing_stop", time_in_force="day", trail_percent=5.0,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["type"] == "trailing_stop"
    assert payload["trail_percent"] == "5.0"
    assert "trail_price" not in payload


@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_bracket_payload_includes_nested_legs(mock_post):
    mock_post.return_value = _response(200, {"id": "o1"})
    submit_order(
        "K", "S", client_order_id="c1", ticker="AAPL", side="buy", qty=10,
        order_type="market", time_in_force="day", order_class="bracket",
        take_profit_price=120.0, stop_loss_price=90.0, stop_loss_limit_price=89.5,
    )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"] == {"limit_price": "120.0"}
    assert payload["stop_loss"] == {"stop_price": "90.0", "limit_price": "89.5"}


@patch("services.alpaca_trading_client.httpx.post")
def test_submit_order_without_new_fields_omits_them_from_payload(mock_post):
    """Regression guard: Stage 1 market/limit callers (the trading agent,
    existing tests) must see an unchanged payload shape."""
    mock_post.return_value = _response(200, {"id": "o1"})
    submit_order(
        "K", "S", client_order_id="c1", ticker="AAPL", side="buy", qty=10,
        order_type="market", time_in_force="day",
    )
    payload = mock_post.call_args.kwargs["json"]
    for key in ("stop_price", "trail_price", "trail_percent", "order_class", "take_profit", "stop_loss", "limit_price"):
        assert key not in payload
