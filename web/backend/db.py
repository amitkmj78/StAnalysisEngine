import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

_user_pool: asyncpg.Pool | None = None
_service_pool: asyncpg.Pool | None = None
_crawlsearch_pool: asyncpg.Pool | None = None


async def init_pools() -> None:
    global _user_pool, _service_pool, _crawlsearch_pool
    _user_pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=10)
    _service_pool = await asyncpg.create_pool(os.environ["DATABASE_URL_SERVICE"], min_size=1, max_size=5)

    # CrawlSearch (../CrawlSearch) is a separate, optional product with its
    # own Postgres database — only pooled when configured, so environments
    # that don't run it (e.g. CI) aren't required to set this.
    crawlsearch_url = os.getenv("CRAWLSEARCH_DATABASE_URL")
    if crawlsearch_url:
        _crawlsearch_pool = await asyncpg.create_pool(crawlsearch_url, min_size=1, max_size=2)


async def close_pools() -> None:
    if _user_pool is not None:
        await _user_pool.close()
    if _service_pool is not None:
        await _service_pool.close()
    if _crawlsearch_pool is not None:
        await _crawlsearch_pool.close()


def crawlsearch_configured() -> bool:
    return _crawlsearch_pool is not None


@asynccontextmanager
async def user_conn(user_id: str) -> AsyncIterator[asyncpg.Connection]:
    """
    Connection scoped to one request's user. Runs inside a transaction with
    `app.user_id` set as a session var for the duration of that transaction
    only (SET LOCAL) — every RLS policy in the schema keys off this, so a
    bug in a router's SQL can't leak another user's rows the way a missed
    `WHERE user_id = ...` clause could with app-level filtering alone.
    """
    assert _user_pool is not None, "init_pools() must run before user_conn()"
    async with _user_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
            yield conn


@asynccontextmanager
async def service_conn() -> AsyncIterator[asyncpg.Connection]:
    """
    Bookkeeping connection (request_log daily quota) via the BYPASSRLS
    app_service role — mirrors the old Supabase service_role trust model:
    backend-owned writes, not scoped to a single user's RLS policy.
    """
    assert _service_pool is not None, "init_pools() must run before service_conn()"
    async with _service_pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def crawlsearch_conn() -> AsyncIterator[asyncpg.Connection]:
    """
    Connection to CrawlSearch's own Postgres database — a separate
    product with its own schema (pages, seen_urls), not part of this
    app's own tables. Used only by the admin SQL console
    (web/backend/routers/admin_sql.py) so CrawlSearch's index can be
    browsed/queried alongside stanalysisengine's own tables in one
    place, without giving every other part of this app a dependency on
    CrawlSearch being deployed.
    """
    assert _crawlsearch_pool is not None, "CRAWLSEARCH_DATABASE_URL not configured"
    async with _crawlsearch_pool.acquire() as conn:
        yield conn
