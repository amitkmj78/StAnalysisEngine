from datetime import datetime, time

from services.notification_dispatcher import DEFAULT_PREFERENCE, is_within_quiet_hours


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 9, 30, hour, minute)


def test_default_preference_is_enabled_email_and_inapp():
    assert DEFAULT_PREFERENCE == {"enabled": True, "channel_email": True, "channel_inapp": True}


def test_no_quiet_hours_configured_is_never_quiet():
    assert is_within_quiet_hours(_at(3), None, None) is False
    assert is_within_quiet_hours(_at(3), time(22, 0), None) is False
    assert is_within_quiet_hours(_at(3), None, time(7, 0)) is False


def test_same_day_window_inside_and_outside():
    start, end = time(9, 0), time(17, 0)
    assert is_within_quiet_hours(_at(12), start, end) is True
    assert is_within_quiet_hours(_at(9), start, end) is True  # inclusive start
    assert is_within_quiet_hours(_at(17), start, end) is False  # exclusive end
    assert is_within_quiet_hours(_at(8, 59), start, end) is False
    assert is_within_quiet_hours(_at(20), start, end) is False


def test_overnight_wraparound_window():
    # 22:00 -> 07:00 spans midnight.
    start, end = time(22, 0), time(7, 0)
    assert is_within_quiet_hours(_at(23), start, end) is True
    assert is_within_quiet_hours(_at(2), start, end) is True  # after midnight, still quiet
    assert is_within_quiet_hours(_at(6, 59), start, end) is True
    assert is_within_quiet_hours(_at(7, 0), start, end) is False  # exclusive end
    assert is_within_quiet_hours(_at(12), start, end) is False  # midday, not quiet
    assert is_within_quiet_hours(_at(21, 59), start, end) is False


def test_zero_width_window_is_never_quiet():
    # start == end: every hour satisfies "start <= end" (same-day branch)
    # but the half-open [start, end) interval is empty.
    same = time(9, 0)
    assert is_within_quiet_hours(_at(9), same, same) is False
    assert is_within_quiet_hours(_at(12), same, same) is False
