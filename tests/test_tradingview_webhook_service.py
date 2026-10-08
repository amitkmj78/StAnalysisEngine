import pytest

from services.tradingview_webhook_service import TradingViewAlertError, parse_tradingview_alert


def test_parse_json_payload_with_ticker_price_and_message():
    body = b'{"ticker": "aapl", "price": 150.25, "message": "Buy signal triggered"}'
    result = parse_tradingview_alert(body)
    assert result == {"ticker": "AAPL", "price": 150.25, "direction": "LONG", "message": "Buy signal triggered"}


def test_parse_accepts_symbol_and_close_as_synonyms():
    body = b'{"symbol": "MSFT", "close": 420.5, "comment": "Sell: overbought"}'
    result = parse_tradingview_alert(body)
    assert result["ticker"] == "MSFT"
    assert result["price"] == 420.5
    assert result["direction"] == "SHORT"


def test_parse_direction_from_explicit_action_field():
    body = b'{"ticker": "TSLA", "action": "sell", "message": "no direction words here"}'
    assert parse_tradingview_alert(body)["direction"] == "SHORT"


def test_parse_direction_none_when_unidentifiable():
    body = b'{"ticker": "NVDA", "message": "price crossed the moving average"}'
    assert parse_tradingview_alert(body)["direction"] is None


def test_parse_raises_when_no_ticker_field_present():
    body = b'{"message": "Something happened, but which stock?"}'
    with pytest.raises(TradingViewAlertError):
        parse_tradingview_alert(body)


def test_parse_raises_for_plain_text_body_with_no_ticker():
    body = b"just a plain text alert, no JSON at all"
    with pytest.raises(TradingViewAlertError):
        parse_tradingview_alert(body)


def test_parse_handles_malformed_json_gracefully_as_no_ticker():
    body = b'{"ticker": "AAPL" not valid json'
    with pytest.raises(TradingViewAlertError):
        parse_tradingview_alert(body)


def test_parse_missing_price_is_none_not_zero():
    body = b'{"ticker": "AAPL", "message": "just a note"}'
    result = parse_tradingview_alert(body)
    assert result["price"] is None


def test_parse_non_numeric_price_is_none_not_an_error():
    body = b'{"ticker": "AAPL", "price": "n/a"}'
    result = parse_tradingview_alert(body)
    assert result["price"] is None
