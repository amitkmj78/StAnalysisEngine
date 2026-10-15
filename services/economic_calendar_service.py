"""
Economic calendar awareness banner: known, publicly-scheduled macro data
releases surfaced a few days in advance -- "a market-moving data release
is coming," never a directional call. Unlike services/market_internals_
service.py's Market Direction score, this isn't a prediction of anything,
so none of that module's validation-gate machinery applies here: these are
dates the government has already announced, the same way a calendar app
isn't "validated" for correctly claiming tomorrow is Tuesday.

CPI, PPI, GDP, Retail Sales and the Jobs Report (nonfarm payrolls) are all
standard BLS/BEA/Census releases with stable FRED release IDs (verified
live against https://fred.stlouisfed.org/release?rid=<id> while building
this, not copied from a hardcoded list found in old docs/packages, which
turned out to be unreliable). FOMC meeting dates are deliberately NOT
fetched from FRED: FRED does carry an "FOMC Press Release" release (rid
101), but its release page shows only past dates, with nothing indicating
reliable forward-scheduled coverage the way the BLS/BEA releases have --
so FOMC is sourced only from admin-entered rows in the
economic_event_overrides table (the Fed publishes a full year of meeting
dates at once, so this is a few minutes of data entry once or twice a
year), which also serves as a manual corrective for any other type FRED
misses.

Deliberately split from the HTTP-fetch half (fails open, hits a real
external API, worth mocking out entirely in tests) and the merge/window
logic (pure, given FRED dates + override rows as plain data) -- see
get_upcoming_economic_events's own docstring.
"""

import logging
import os
from datetime import date, datetime, timedelta

import httpx

from .cache_utils import ttl_cache

logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred"
REQUEST_TIMEOUT_SECONDS = 15.0
# This data changes at most once a day (a new scheduled date gets added to
# the government's own calendar), so a day-long cache never shows stale
# information a user could act differently on.
RELEASE_DATES_TTL_SECONDS = 60 * 60 * 24

# event_type -> FRED release_id. FOMC is intentionally absent -- see module
# docstring.
EVENT_TYPE_RELEASE_IDS: dict[str, int] = {
    "CPI": 10,
    "RETAIL_SALES": 9,
    "PPI": 46,
    "JOBS_REPORT": 50,
    "GDP": 53,
}

# Plain display title per type, used when FRED is the source (an admin
# override row carries its own title instead).
EVENT_TYPE_TITLES: dict[str, str] = {
    "CPI": "CPI Report",
    "RETAIL_SALES": "Retail Sales Report",
    "PPI": "PPI Report",
    "JOBS_REPORT": "Jobs Report (Nonfarm Payrolls)",
    "GDP": "GDP Report",
    "FOMC": "FOMC Meeting",
}

DISCLOSURE = (
    "Awareness only -- these are known, publicly-scheduled data release "
    "dates, not a prediction of what they'll show or how markets will "
    "react. Not a buy or sell signal."
)


@ttl_cache(maxsize=16, ttl_seconds=RELEASE_DATES_TTL_SECONDS)
def fetch_upcoming_release_dates(release_id: int) -> list[date]:
    """Forward-scheduled dates (today or later) for one FRED release, via
    fred/release/dates with include_release_dates_with_no_data=true (that
    flag is what includes dates the release hasn't happened for yet --
    without it, FRED only returns dates data has actually been published
    for). Fails open to [] on a missing key or any HTTP/parse error --
    same "never let a broken external dependency break the page"
    convention as this codebase's send_*_email functions -- so a FRED
    outage just means those event types are temporarily missing from the
    banner, not a broken page."""
    if not FRED_API_KEY:
        logger.warning("economic_calendar: FRED_API_KEY is not configured, skipping release_id=%s", release_id)
        return []
    try:
        response = httpx.get(
            f"{FRED_BASE_URL}/release/dates",
            params={
                "release_id": release_id,
                "realtime_start": date.today().isoformat(),
                "include_release_dates_with_no_data": "true",
                "api_key": FRED_API_KEY,
                "file_type": "json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            datetime.strptime(row["date"], "%Y-%m-%d").date()
            for row in payload.get("release_dates", [])
        ]
    except Exception as e:
        logger.warning("economic_calendar: failed to fetch release_id=%s: %s", release_id, e)
        return []


def get_upcoming_economic_events(overrides: list[dict], window_days: int = 7) -> list[dict]:
    """Merges FRED-sourced dates for every type in EVENT_TYPE_RELEASE_IDS
    with `overrides` (admin-entered rows -- the only source for FOMC, and
    a manual corrective for any other type), dedupes same type+date (an
    override's title/note wins over FRED's generic title on a collision),
    and filters to [today, today + window_days]. Pure given its inputs --
    fetch_upcoming_release_dates is the only part that touches the
    network, so tests pass `overrides` directly and monkeypatch that
    function rather than hitting real FRED.

    `overrides`: [{"event_type", "event_date": date, "title", "note"}, ...]
    Returns: [{"event_type", "title", "event_date": "YYYY-MM-DD",
    "days_until", "source": "fred" | "admin", "note"}, ...], sorted by date.
    """
    today = date.today()
    window_end = today + timedelta(days=window_days)

    events_by_key: dict[tuple[str, date], dict] = {}

    for event_type, release_id in EVENT_TYPE_RELEASE_IDS.items():
        for event_date in fetch_upcoming_release_dates(release_id):
            if today <= event_date <= window_end:
                events_by_key[(event_type, event_date)] = {
                    "event_type": event_type,
                    "title": EVENT_TYPE_TITLES[event_type],
                    "event_date": event_date,
                    "source": "fred",
                    "note": None,
                }

    for row in overrides:
        event_date = row["event_date"]
        if today <= event_date <= window_end:
            key = (row["event_type"], event_date)
            events_by_key[key] = {
                "event_type": row["event_type"],
                "title": row["title"] or EVENT_TYPE_TITLES.get(row["event_type"], row["event_type"]),
                "event_date": event_date,
                "source": "admin",
                "note": row.get("note"),
            }

    events = sorted(events_by_key.values(), key=lambda e: e["event_date"])
    return [
        {
            "event_type": e["event_type"],
            "title": e["title"],
            "event_date": e["event_date"].isoformat(),
            "days_until": (e["event_date"] - today).days,
            "source": e["source"],
            "note": e["note"],
        }
        for e in events
    ]
