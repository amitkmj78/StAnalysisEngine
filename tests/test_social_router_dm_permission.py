import asyncio

from web.backend.routers.social import _dm_allowed


class _FakeConn:
    def __init__(self, follows: bool, allowed: bool):
        self._follows = follows
        self._allowed = allowed

    async def fetchval(self, sql, *args):
        if "author_follows" in sql:
            return 1 if self._follows else None
        if "dm_allowed_senders" in sql:
            return 1 if self._allowed else None
        raise AssertionError(f"unexpected query: {sql}")


def test_dm_allowed_when_recipient_follows_sender():
    conn = _FakeConn(follows=True, allowed=False)
    assert asyncio.run(_dm_allowed(conn, "sender", "recipient")) is True


def test_dm_allowed_when_recipient_explicitly_allowed_sender():
    conn = _FakeConn(follows=False, allowed=True)
    assert asyncio.run(_dm_allowed(conn, "sender", "recipient")) is True


def test_dm_denied_when_neither_follow_nor_allow_list():
    conn = _FakeConn(follows=False, allowed=False)
    assert asyncio.run(_dm_allowed(conn, "sender", "recipient")) is False
