"""SCAN-2: point-in-time S&P 500 membership.

Membership on any date is today's list, with each later change undone: a stock added after that date was not a member
then, and a stock removed after it was. The changes come from Wikipedia's history of S&P 500 components. That source is
free but not licensed, and it lists tickers, not prices: a removed stock that yfinance has no price history for can't be
tested, and the scan skips it and says so.
"""

import io
import logging

import pandas as pd
import requests

from .cache_utils import ttl_cache
from .stock_finder_service import fetch_sp500_tickers

logger = logging.getLogger(__name__)

CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; StAnalysisEngine/1.0)"}


def _norm(symbol) -> str | None:
    """Wikipedia writes BRK.B; yfinance writes BRK-B. Both are stored in the yfinance form."""
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    return symbol.strip().upper().replace(".", "-")


@ttl_cache(maxsize=2, ttl_seconds=60 * 60 * 24)
def fetch_changes() -> pd.DataFrame:
    """Every S&P 500 addition and removal, oldest first: date, added (or None), removed (or None)."""
    response = requests.get(CHANGES_URL, headers=USER_AGENT, timeout=20)
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text), flavor="lxml")[0]
    changes = pd.DataFrame({
        "date": pd.to_datetime(table.iloc[:, 0], errors="coerce"),
        "added": table.iloc[:, 1].map(_norm),
        "removed": table.iloc[:, 3].map(_norm),
    })
    return changes.dropna(subset=["date"]).sort_values("date", kind="stable").reset_index(drop=True)


@ttl_cache(maxsize=2, ttl_seconds=60 * 60 * 24)
def current_members() -> frozenset:
    return frozenset(_norm(t) for t in fetch_sp500_tickers() if _norm(t))


def members_on(day) -> set[str]:
    """Who was in the S&P 500 at the close of `day`."""
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)
    members = set(current_members())
    changes = fetch_changes()
    later = changes[changes["date"] > day]
    for _, row in later.iloc[::-1].iterrows():
        # A blank cell in the table comes through as NaN, which is truthy in Python: only real ticker strings count.
        if isinstance(row["added"], str):
            members.discard(row["added"])
        if isinstance(row["removed"], str):
            members.add(row["removed"])
    return members


def removed_after(day) -> list[str]:
    """Tickers that left the index after `day`, for the results' disclosure."""
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)
    changes = fetch_changes()
    later = changes[(changes["date"] > day) & changes["removed"].map(lambda x: isinstance(x, str))]
    return sorted(set(later["removed"]))


def member_flags(ticker: str, index: pd.DatetimeIndex, start_members: set[str] | None = None) -> pd.Series:
    """True on each session of `index` when `ticker` was an S&P 500 member at that session's close."""
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()
    if len(dates) == 0:
        return pd.Series(dtype=bool)
    state = ticker in (start_members if start_members is not None else members_on(dates[0]))
    changes = fetch_changes()
    events = changes[((changes["added"] == ticker) | (changes["removed"] == ticker))
                     & (changes["date"] > dates[0]) & (changes["date"] <= dates[-1])]
    flags = []
    pointer = 0
    event_rows = list(events.itertuples(index=False))
    for day in dates:
        while pointer < len(event_rows) and event_rows[pointer].date <= day:
            state = event_rows[pointer].added == ticker
            pointer += 1
        flags.append(state)
    return pd.Series(flags, index=index)
