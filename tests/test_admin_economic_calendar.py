"""Admin CRUD for economic_event_overrides, and the public economic-
calendar endpoint's available:false-when-off convention. Same fake-
connection pattern as tests/test_admin_roles_and_theme.py."""

import asyncio
from datetime import date

import asyncpg
import pytest
from fastapi import HTTPException

import web.backend.routers.admin_economic_calendar as admin_cal_mod
import web.backend.routers.economic_calendar as public_cal_mod
from web.backend.routers.admin_economic_calendar import (
    EconomicEventOverrideRequest,
    create_economic_event_override,
    delete_economic_event_override,
)
from web.backend.routers.economic_calendar import get_economic_calendar


class _FakeConn:
    def __init__(self, fetch=None, fetchrow=None):
        self._fetch = fetch or (lambda sql, *a: [])
        self._fetchrow = fetchrow or (lambda sql, *a: None)

    async def fetch(self, sql, *args):
        return self._fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        return self._fetchrow(sql, *args)


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def test_create_override_rejects_an_unknown_event_type():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            create_economic_event_override(
                EconomicEventOverrideRequest(event_type="NOT_A_TYPE", event_date=date(2026, 12, 1), title="x")
            )
        )
    assert exc.value.status_code == 422


def test_create_override_inserts_and_returns_the_row(monkeypatch):
    row = {"id": 1, "event_type": "FOMC", "event_date": date(2026, 12, 17), "title": "FOMC Decision", "note": None}
    conn = _FakeConn(fetchrow=lambda sql, *a: row)
    monkeypatch.setattr(admin_cal_mod, "service_conn", lambda: _FakeConnCtx(conn))

    result = asyncio.run(
        create_economic_event_override(
            EconomicEventOverrideRequest(event_type="FOMC", event_date=date(2026, 12, 17), title="FOMC Decision")
        )
    )
    assert result["id"] == "1"
    assert result["event_type"] == "FOMC"
    assert result["event_date"] == "2026-12-17"


def test_create_override_rejects_a_duplicate_type_and_date(monkeypatch):
    def fetchrow(sql, *a):
        raise asyncpg.UniqueViolationError("duplicate")

    conn = _FakeConn(fetchrow=fetchrow)
    monkeypatch.setattr(admin_cal_mod, "service_conn", lambda: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            create_economic_event_override(
                EconomicEventOverrideRequest(event_type="CPI", event_date=date(2026, 11, 1), title="CPI Report")
            )
        )
    assert exc.value.status_code == 409


def test_delete_override_404s_when_not_found(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: None)
    monkeypatch.setattr(admin_cal_mod, "service_conn", lambda: _FakeConnCtx(conn))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(delete_economic_event_override("999"))
    assert exc.value.status_code == 404


def test_public_endpoint_unavailable_when_toggle_is_off(monkeypatch):
    async def disabled(key, default):
        return False

    monkeypatch.setattr(public_cal_mod, "get_setting_bool", disabled)
    result = asyncio.run(get_economic_calendar())
    assert result == {"available": False, "reason": "economic_calendar_enabled is off"}


def test_public_endpoint_returns_upcoming_events_when_enabled(monkeypatch):
    async def enabled(key, default):
        return True

    conn = _FakeConn(fetch=lambda sql, *a: [])
    monkeypatch.setattr(public_cal_mod, "get_setting_bool", enabled)
    monkeypatch.setattr(public_cal_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(public_cal_mod, "get_upcoming_economic_events", lambda overrides: [{"event_type": "CPI"}])

    result = asyncio.run(get_economic_calendar())
    assert result["available"] is True
    assert result["upcoming"] == [{"event_type": "CPI"}]
    assert "disclosure" in result
