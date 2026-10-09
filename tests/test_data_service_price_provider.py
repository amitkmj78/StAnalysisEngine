from unittest.mock import patch

import pytest

from services import data_service
from services.alpaca_client import AlpacaSymbolNotFound
from services.data_service import (
    get_effective_price,
    get_extended_hours_price,
    get_latest_price,
    get_previous_close,
)
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


def test_get_effective_price_prefers_extended_hours_when_available():
    with patch(
        "services.data_service.get_extended_hours_price",
        return_value={"state": "POST", "price": 145.0, "change_pct": 1.2},
    ), patch("services.data_service.get_latest_price", return_value=142.0) as mock_latest:
        price = get_effective_price("AFTER_HOURS_TICKER")
    assert price == 145.0
    mock_latest.assert_not_called()


def test_get_effective_price_falls_back_to_regular_price_outside_extended_hours():
    with patch("services.data_service.get_extended_hours_price", return_value=None), patch(
        "services.data_service.get_latest_price", return_value=142.0
    ):
        price = get_effective_price("REGULAR_HOURS_TICKER")
    assert price == 142.0


def test_get_previous_close_reads_fast_info():
    mock_ticker = type("T", (), {"fast_info": {"previousClose": 141.5}})()
    with patch("services.data_service.yf.Ticker", return_value=mock_ticker):
        prev_close = get_previous_close("PREV_CLOSE_TICKER_1")
    assert prev_close == 141.5


def test_get_previous_close_returns_none_when_missing():
    mock_ticker = type("T", (), {"fast_info": {}})()
    with patch("services.data_service.yf.Ticker", return_value=mock_ticker):
        assert get_previous_close("PREV_CLOSE_TICKER_2") is None


def test_get_previous_close_briefly_caches_a_failed_lookup_then_retries():
    """Regression test, two incidents deep. First: a rate-limited/failed
    fetch must not be remembered for the same 1hr TTL as a real result —
    production hit exactly this, where a transient Yahoo rate-limit
    blanked Today's Gain/Loss for most of a portfolio for a full hour
    even though Yahoo recovered seconds later. Second (the opposite
    failure mode of literally zero caching): a PERSISTENTLY-failing
    ticker (e.g. a mutual fund neither Yahoo nor Alpaca can quote) was
    then observed retrying its full backoff chain on every single poll
    of a page refreshing every ~10-20s, forever — real, repeated latency
    with zero chance of ever succeeding. The fix for both: a failure IS
    cached, just much more briefly (_PREVIOUS_CLOSE_FAILURE_TTL_SECONDS)
    than a real result (_PREVIOUS_CLOSE_TTL_SECONDS)."""
    with patch("services.data_service.fetch_with_backoff", side_effect=Exception("Too Many Requests")), patch(
        "services.data_service.get_alpaca_previous_close", return_value=None
    ):
        assert get_previous_close("PREV_CLOSE_TICKER_3") is None

    # Immediately calling again must NOT retry -- the short failure-TTL
    # cache should serve the same None without a second fetch attempt.
    with patch("services.data_service.fetch_with_backoff", return_value=99.5) as mock_fetch:
        assert get_previous_close("PREV_CLOSE_TICKER_3") is None
        mock_fetch.assert_not_called()

    # Once the short failure TTL has elapsed, the next call retries for real.
    data_service._previous_close_cache_ts["PREV_CLOSE_TICKER_3"] -= data_service._PREVIOUS_CLOSE_FAILURE_TTL_SECONDS + 1
    with patch("services.data_service.fetch_with_backoff", return_value=99.5):
        assert get_previous_close("PREV_CLOSE_TICKER_3") == 99.5


def test_get_previous_close_caches_a_successful_lookup():
    """The other half of the same fix: a real result IS cached (it's
    static for the trading day) -- a second call shouldn't refetch."""
    with patch("services.data_service.fetch_with_backoff", return_value=88.25) as mock_fetch:
        assert get_previous_close("PREV_CLOSE_TICKER_4") == 88.25
        assert get_previous_close("PREV_CLOSE_TICKER_4") == 88.25
    assert mock_fetch.call_count == 1


def test_get_previous_close_falls_back_to_alpaca_when_yahoo_fails():
    """A real mitigation for today's incident: when Yahoo fails, this
    tries Alpaca's daily bars before giving up -- and a result served
    from that fallback is cached at the full (not the short-failure) TTL,
    since it's a real answer, not a miss."""
    with patch("services.data_service.fetch_with_backoff", side_effect=Exception("Too Many Requests")), patch(
        "services.data_service.get_alpaca_previous_close", return_value=77.10
    ) as mock_alpaca:
        assert get_previous_close("PREV_CLOSE_TICKER_5") == 77.10
    mock_alpaca.assert_called_once_with("PREV_CLOSE_TICKER_5")

    # Cached at the full TTL -- a second call within it must not refetch
    # from either source.
    with patch("services.data_service.fetch_with_backoff") as mock_fetch, patch(
        "services.data_service.get_alpaca_previous_close"
    ) as mock_alpaca_2:
        assert get_previous_close("PREV_CLOSE_TICKER_5") == 77.10
    mock_fetch.assert_not_called()
    mock_alpaca_2.assert_not_called()


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
