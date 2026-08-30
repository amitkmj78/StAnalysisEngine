import pytest

from services.price_provider import get_price_provider, set_price_provider


@pytest.fixture(autouse=True)
def _reset_provider():
    set_price_provider("yahoo")
    yield
    set_price_provider("yahoo")


def test_defaults_to_yahoo():
    assert get_price_provider() == "yahoo"


def test_set_price_provider_switches_to_alpaca():
    set_price_provider("alpaca")
    assert get_price_provider() == "alpaca"


def test_set_price_provider_rejects_unknown_provider():
    with pytest.raises(ValueError):
        set_price_provider("robinhood")
    assert get_price_provider() == "yahoo"
