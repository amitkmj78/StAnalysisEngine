"""services/economic_calendar_service.py's merge/dedupe/window-filter
logic. fetch_upcoming_release_dates (the only part that touches the
network) is always monkeypatched -- these tests never hit real FRED."""

from datetime import date, timedelta

import services.economic_calendar_service as cal_mod
from services.economic_calendar_service import get_upcoming_economic_events

TODAY = date.today()


def _fred_dates_for(mapping: dict[int, list[date]]):
    def fake(release_id: int) -> list[date]:
        return mapping.get(release_id, [])
    return fake


def test_includes_fred_sourced_dates_within_the_window(monkeypatch):
    cpi_release_id = cal_mod.EVENT_TYPE_RELEASE_IDS["CPI"]
    monkeypatch.setattr(
        cal_mod, "fetch_upcoming_release_dates", _fred_dates_for({cpi_release_id: [TODAY + timedelta(days=2)]})
    )
    events = get_upcoming_economic_events(overrides=[], window_days=7)
    assert len(events) == 1
    assert events[0]["event_type"] == "CPI"
    assert events[0]["source"] == "fred"
    assert events[0]["days_until"] == 2
    assert events[0]["title"] == "CPI Report"


def test_excludes_dates_outside_the_window(monkeypatch):
    cpi_release_id = cal_mod.EVENT_TYPE_RELEASE_IDS["CPI"]
    monkeypatch.setattr(
        cal_mod,
        "fetch_upcoming_release_dates",
        _fred_dates_for({cpi_release_id: [TODAY + timedelta(days=30), TODAY - timedelta(days=1)]}),
    )
    events = get_upcoming_economic_events(overrides=[], window_days=7)
    assert events == []


def test_admin_override_is_the_only_source_for_fomc(monkeypatch):
    monkeypatch.setattr(cal_mod, "fetch_upcoming_release_dates", _fred_dates_for({}))
    overrides = [{"event_type": "FOMC", "event_date": TODAY + timedelta(days=1), "title": "FOMC Decision", "note": None}]
    events = get_upcoming_economic_events(overrides=overrides, window_days=7)
    assert len(events) == 1
    assert events[0]["event_type"] == "FOMC"
    assert events[0]["source"] == "admin"
    assert events[0]["title"] == "FOMC Decision"


def test_admin_override_wins_on_same_type_and_date_as_a_fred_result(monkeypatch):
    cpi_release_id = cal_mod.EVENT_TYPE_RELEASE_IDS["CPI"]
    event_date = TODAY + timedelta(days=3)
    monkeypatch.setattr(cal_mod, "fetch_upcoming_release_dates", _fred_dates_for({cpi_release_id: [event_date]}))
    overrides = [{"event_type": "CPI", "event_date": event_date, "title": "CPI Report (corrected)", "note": "fixed"}]
    events = get_upcoming_economic_events(overrides=overrides, window_days=7)
    assert len(events) == 1
    assert events[0]["source"] == "admin"
    assert events[0]["title"] == "CPI Report (corrected)"
    assert events[0]["note"] == "fixed"


def test_results_are_sorted_by_date(monkeypatch):
    cpi_release_id = cal_mod.EVENT_TYPE_RELEASE_IDS["CPI"]
    ppi_release_id = cal_mod.EVENT_TYPE_RELEASE_IDS["PPI"]
    monkeypatch.setattr(
        cal_mod,
        "fetch_upcoming_release_dates",
        _fred_dates_for({cpi_release_id: [TODAY + timedelta(days=5)], ppi_release_id: [TODAY + timedelta(days=1)]}),
    )
    events = get_upcoming_economic_events(overrides=[], window_days=7)
    assert [e["event_type"] for e in events] == ["PPI", "CPI"]


def test_fetch_upcoming_release_dates_fails_open_without_an_api_key(monkeypatch):
    # Distinct release_id per test below -- fetch_upcoming_release_dates is
    # @ttl_cache'd on its args, so reusing the same id across tests in this
    # process would hit a previous test's cached result instead of
    # exercising this test's own patched path.
    monkeypatch.setattr(cal_mod, "FRED_API_KEY", None)
    assert cal_mod.fetch_upcoming_release_dates(9001) == []


def test_fetch_upcoming_release_dates_fails_open_on_http_error(monkeypatch):
    monkeypatch.setattr(cal_mod, "FRED_API_KEY", "fake-key")

    def boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(cal_mod.httpx, "get", boom)
    assert cal_mod.fetch_upcoming_release_dates(9002) == []
