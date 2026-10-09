from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from services.social_chat_service import EASTERN, is_market_hours_now, validate_room


def test_market_hours_during_session():
    # Wednesday, 10:00 ET.
    assert is_market_hours_now(datetime(2026, 10, 7, 10, 0, tzinfo=EASTERN)) is True


def test_market_hours_before_open():
    assert is_market_hours_now(datetime(2026, 10, 7, 9, 0, tzinfo=EASTERN)) is False


def test_market_hours_at_close_is_closed():
    assert is_market_hours_now(datetime(2026, 10, 7, 16, 0, tzinfo=EASTERN)) is False


def test_market_hours_weekend_is_closed():
    # Saturday.
    assert is_market_hours_now(datetime(2026, 10, 10, 10, 0, tzinfo=EASTERN)) is False


def test_market_hours_converts_other_timezones():
    utc_time = datetime(2026, 10, 7, 14, 0, tzinfo=ZoneInfo("UTC"))  # 10:00 ET
    assert is_market_hours_now(utc_time) is True


def test_validate_room_general_case_insensitive():
    assert validate_room("General") == "general"
    assert validate_room(" general ") == "general"


def test_validate_room_valid_ticker_is_uppercased():
    assert validate_room("aapl") == "AAPL"


def test_validate_room_rejects_invalid_ticker():
    with pytest.raises(ValueError):
        validate_room("not a ticker!!")
