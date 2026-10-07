from services.alpaca_client import ALPACA_PERIOD_DAYS

# Every history period the app asks for. A period missing here makes Alpaca return no history at all.
PERIODS_USED_BY_THE_APP = ("1d", "5d", "7d", "60d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "730d", "5y", "10y", "max")


def test_every_period_the_app_requests_is_known_to_the_alpaca_reader():
    missing = [p for p in PERIODS_USED_BY_THE_APP if p not in ALPACA_PERIOD_DAYS]
    assert missing == []
