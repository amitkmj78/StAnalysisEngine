from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from web.backend.db import service_conn

DAILY_QUOTA = 200


def _rate_limit_key(request: Request) -> str:
    """
    Key burst-limiting by the authenticated user when available (auth
    dependency has already run for protected routes), falling back to
    remote address for anything unauthenticated.
    """
    user = getattr(request.state, "user", None)
    return user["id"] if user else get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


async def enforce_daily_quota(request: Request, endpoint: str) -> None:
    """
    Durable per-user daily quota backed by Postgres (survives process
    restarts, unlike the in-memory slowapi burst limiter). Call after the
    auth dependency has populated request.state.user. Deliberately generic
    (free-text `endpoint`) so LLM endpoints can reuse this same mechanism
    later without a schema change.
    """
    user = request.state.user
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    async with service_conn() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM request_log WHERE user_id = $1::uuid AND created_at >= $2",
            user["id"],
            since,
        )
        if count >= DAILY_QUOTA:
            raise HTTPException(429, "Daily request limit reached")

        await conn.execute(
            "INSERT INTO request_log (user_id, endpoint) VALUES ($1::uuid, $2)",
            user["id"],
            endpoint,
        )
