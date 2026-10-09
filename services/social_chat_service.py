"""SOC-7: pure helpers for the polling-based chat rooms. No WebSocket
layer exists anywhere in this app -- every other "live" feature (the
header market ticker, the alerts inbox, price badges) polls on an
interval, and this follows the same posture rather than introducing a
new connection/broadcast layer for one feature.
"""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN_MINUTES = 9 * 60 + 30
MARKET_CLOSE_MINUTES = 16 * 60

GENERAL_ROOM = "general"
_TICKER_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")


def is_market_hours_now(now: datetime | None = None) -> bool:
    """Mon-Fri 9:30-16:00 America/New_York -- same day-of-week
    convention web/backend/scheduler.py's CronTriggers already use.
    Does NOT account for market holidays (disclosed gap, not a full
    market calendar)."""
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return MARKET_OPEN_MINUTES <= minutes < MARKET_CLOSE_MINUTES


def validate_room(raw_room: str) -> str:
    """A room is either 'general' or a valid ticker -- no ad-hoc room
    creation (a disclosed scope cut from SOC-7's literal text)."""
    normalized = raw_room.strip()
    if normalized.lower() == GENERAL_ROOM:
        return GENERAL_ROOM
    ticker = normalized.upper()
    if not _TICKER_PATTERN.match(ticker):
        raise ValueError("room must be 'general' or a valid ticker")
    return ticker
