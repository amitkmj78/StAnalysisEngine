"""SAF-1/2/3/7: the warn endpoint, the generic content-report endpoint's
validation, and the automated mention-spike / duplicate-post detectors.
Same fake-connection pattern as tests/test_admin_roles_and_theme.py.
"""

import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import services.community_safety_service as safety_mod
import web.backend.routers.admin_users as admin_users_mod
import web.backend.routers.social as social_mod
from services.community_safety_service import (
    DUPLICATE_WINDOW_HOURS,
    MENTION_SPIKE_MIN_TODAY,
    MENTION_SPIKE_RATIO,
    SMALL_CAP_CEILING_USD,
    detect_duplicate_posts,
    detect_mention_spikes,
)
from web.backend.routers.admin_users import WarnRequest, warn_user
from web.backend.routers.social import ContentReportRequest, report_content


class _FakeConn:
    def __init__(self, fetch=None, fetchval=None, fetchrow=None):
        self._fetch = fetch or (lambda sql, *a: [])
        self._fetchval = fetchval or (lambda sql, *a: None)
        self._fetchrow = fetchrow or (lambda sql, *a: None)
        self.executed: list[tuple] = []

    async def fetch(self, sql, *args):
        return self._fetch(sql, *args)

    async def fetchval(self, sql, *args):
        return self._fetchval(sql, *args)

    async def fetchrow(self, sql, *args):
        self.executed.append((sql, args))
        return self._fetchrow(sql, *args)

    async def execute(self, sql, *args):
        self.executed.append((sql, args))
        return "OK"


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *a):
        return False


def _make_request(user_id: str) -> Request:
    # slowapi's @limiter.limit decorator requires a real starlette Request
    # (it reads request.client/.headers for the rate-limit key), not a
    # bare stand-in -- a minimal ASGI scope is enough to construct one.
    scope = {
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "client": ("127.0.0.1", 0), "server": ("testserver", 80), "scheme": "http", "query_string": b"",
    }
    request = Request(scope)
    request.state.user = {"id": user_id}
    return request


# ---------------------------------------------------------------- SAF-1 warn

def test_warn_user_requires_a_nonblank_note():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(warn_user("u1", WarnRequest(note="   ")))
    assert exc.value.status_code == 422


def test_warn_user_404s_for_a_missing_user(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: None)
    monkeypatch.setattr(admin_users_mod, "service_conn", lambda: _FakeConnCtx(conn))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(warn_user("missing", WarnRequest(note="spam")))
    assert exc.value.status_code == 404


def test_warn_user_increments_and_returns_the_new_count(monkeypatch):
    conn = _FakeConn(fetchrow=lambda sql, *a: {"id": "u1", "email": "a@example.com", "warning_count": 2})
    monkeypatch.setattr(admin_users_mod, "service_conn", lambda: _FakeConnCtx(conn))
    result = asyncio.run(warn_user("u1", WarnRequest(note="undisclosed paid promotion")))
    assert result == {"id": "u1", "email": "a@example.com", "warning_count": 2}
    sql, args = conn.executed[0]
    assert "warning_count = warning_count + 1" in sql


# ---------------------------------------------------------------- SAF-2 report

def test_report_content_rejects_an_invalid_content_type():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="idea", content_id="1", reason="spam")))
    assert exc.value.status_code == 422


def test_report_content_rejects_a_blank_reason():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="post", content_id="1", reason="  ")))
    assert exc.value.status_code == 422


def test_report_content_rejects_a_non_numeric_content_id_for_a_post():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="post", content_id="not-a-number", reason="spam")))
    assert exc.value.status_code == 422


def test_report_content_404s_for_a_missing_post(monkeypatch):
    conn = _FakeConn(fetchval=lambda sql, *a: None)
    monkeypatch.setattr(social_mod, "service_conn", lambda: _FakeConnCtx(conn))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="post", content_id="999", reason="spam")))
    assert exc.value.status_code == 404


def test_report_content_auto_hides_a_post_past_the_threshold(monkeypatch):
    values = iter([1, social_mod.AUTO_HIDE_CONTENT_REPORT_THRESHOLD])  # exists check, then report_count
    conn = _FakeConn(fetchval=lambda sql, *a: next(values))
    monkeypatch.setattr(social_mod, "service_conn", lambda: _FakeConnCtx(conn))
    result = asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="post", content_id="42", reason="pump and dump")))
    assert result == {"ok": True, "report_count": social_mod.AUTO_HIDE_CONTENT_REPORT_THRESHOLD, "hidden": True}
    update_calls = [c for c in conn.executed if "UPDATE posts SET hidden" in c[0]]
    assert len(update_calls) == 1


def test_report_content_does_not_hide_below_the_threshold(monkeypatch):
    values = iter([1, 1])  # exists check, then report_count below threshold
    conn = _FakeConn(fetchval=lambda sql, *a: next(values))
    monkeypatch.setattr(social_mod, "service_conn", lambda: _FakeConnCtx(conn))
    result = asyncio.run(report_content(_make_request("u1"), ContentReportRequest(content_type="post", content_id="42", reason="spam")))
    assert result["hidden"] is False


# ---------------------------------------------------------------- SAF-3 detection

def test_detect_mention_spikes_skips_tickers_below_the_minimum_floor(monkeypatch):
    conn = _FakeConn(fetch=lambda sql, *a: [{"ticker": "AAPL", "n": MENTION_SPIKE_MIN_TODAY - 1}] if "created_at::date = $1 GROUP" in sql else [])
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    assert asyncio.run(detect_mention_spikes()) == 0


def test_detect_mention_spikes_flags_a_real_spike_in_a_small_cap(monkeypatch):
    def fetch(sql, *a):
        if "created_at::date = $1 GROUP" in sql:
            return [{"ticker": "SMLL", "n": MENTION_SPIKE_MIN_TODAY * MENTION_SPIKE_RATIO + 5}]
        return [{"ticker": "SMLL", "n": 7}]  # trailing 7 days total -> avg 1/day

    conn = _FakeConn(fetch=fetch)
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(safety_mod, "_get_market_cap", lambda ticker: _async_return(SMALL_CAP_CEILING_USD - 1))
    assert asyncio.run(detect_mention_spikes()) == 1


def test_detect_mention_spikes_ignores_a_large_cap_even_with_a_real_spike(monkeypatch):
    def fetch(sql, *a):
        if "created_at::date = $1 GROUP" in sql:
            return [{"ticker": "BIGCO", "n": 100}]
        return [{"ticker": "BIGCO", "n": 7}]

    conn = _FakeConn(fetch=fetch)
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    monkeypatch.setattr(safety_mod, "_get_market_cap", lambda ticker: _async_return(SMALL_CAP_CEILING_USD * 100))
    assert asyncio.run(detect_mention_spikes()) == 0


def test_detect_duplicate_posts_requires_at_least_two_distinct_authors(monkeypatch):
    rows = [
        {"id": 1, "author_user_id": "author-a", "body": "This is a sufficiently long duplicate post body."},
        {"id": 2, "author_user_id": "author-a", "body": "This is a sufficiently long duplicate post body."},
    ]
    conn = _FakeConn(fetch=lambda sql, *a: rows if "FROM posts WHERE" in sql else [])
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    assert asyncio.run(detect_duplicate_posts()) == 0


def test_detect_duplicate_posts_flags_the_same_body_from_different_authors(monkeypatch):
    rows = [
        {"id": 1, "author_user_id": "author-a", "body": "This is a sufficiently long duplicate post body."},
        {"id": 2, "author_user_id": "author-b", "body": "THIS IS A SUFFICIENTLY LONG DUPLICATE POST BODY."},
    ]
    conn = _FakeConn(fetch=lambda sql, *a: rows if "FROM posts WHERE" in sql else [])
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    assert asyncio.run(detect_duplicate_posts()) == 2


def test_detect_duplicate_posts_ignores_trivially_short_bodies(monkeypatch):
    rows = [
        {"id": 1, "author_user_id": "author-a", "body": "lol"},
        {"id": 2, "author_user_id": "author-b", "body": "lol"},
    ]
    conn = _FakeConn(fetch=lambda sql, *a: rows if "FROM posts WHERE" in sql else [])
    monkeypatch.setattr(safety_mod, "service_conn", lambda: _FakeConnCtx(conn))
    assert asyncio.run(detect_duplicate_posts()) == 0


async def _async_return(value):
    return value


def test_constants_are_sane():
    assert DUPLICATE_WINDOW_HOURS > 0
    assert MENTION_SPIKE_MIN_TODAY > 0
    assert MENTION_SPIKE_RATIO > 1
