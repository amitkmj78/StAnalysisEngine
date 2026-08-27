import os
import time
from datetime import datetime, timezone

import jwt
from fastapi import HTTPException, Request, status

from web.backend.db import service_conn

SESSION_SECRET = os.environ["SESSION_SECRET"]
SESSION_COOKIE_NAME = "session"

# Sessions are otherwise stateless JWTs — nothing re-checks the DB per
# request, so a deactivated/force-logged-out user's already-open tab kept
# working until the token's natural expiry. This cache backs a real-time
# revocation check: session_invalidated_at on `users`, bumped whenever an
# admin deactivates or force-logs-out an account. Cached briefly per user
# (not per request) so revocation is felt within seconds without adding a
# DB round trip to every single authenticated request.
_revocation_cache: dict[str, tuple[float, datetime | None]] = {}
_REVOCATION_CACHE_TTL_SECONDS = 20.0


async def _get_session_invalidated_at(user_id: str) -> datetime | None:
    now = time.monotonic()
    cached = _revocation_cache.get(user_id)
    if cached is not None and now - cached[0] < _REVOCATION_CACHE_TTL_SECONDS:
        return cached[1]
    async with service_conn() as conn:
        value = await conn.fetchval(
            "SELECT session_invalidated_at FROM users WHERE id = $1::uuid", user_id
        )
    _revocation_cache[user_id] = (now, value)
    return value


async def _is_revoked(user_id: str, issued_at: int | None) -> bool:
    invalidated_at = await _get_session_invalidated_at(user_id)
    if invalidated_at is None:
        return False
    if issued_at is None:
        return True
    return datetime.fromtimestamp(issued_at, tz=timezone.utc) < invalidated_at


def _extract_token(request: Request) -> str:
    # Cookie first (how the browser actually authenticates, httpOnly so JS
    # never touches it) — Authorization header kept as a fallback so curl/
    # scripts can still drive the API directly during local dev.
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        return token

    authz = request.headers.get("authorization", "")
    if authz.startswith("Bearer "):
        return authz.removeprefix("Bearer ")

    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing session")


async def verify_bearer_token(request: Request) -> dict:
    token = _extract_token(request)

    try:
        payload = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid session: {exc}")

    if await _is_revoked(payload["sub"], payload.get("iat")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session revoked — please log in again.")

    user = {"id": payload["sub"], "email": payload.get("email")}
    request.state.user = user
    request.state.access_token = token
    return user


async def verify_bearer_token_optional(request: Request) -> dict | None:
    """
    Same as verify_bearer_token but never raises — returns None for a
    missing, invalid, or revoked token instead of a 401. Needed for RS-2's
    tiered content gating: the free tier must stay reachable by a fully
    anonymous visitor, while a signed-in paid subscriber on the same
    endpoint still gets identified. Endpoints that require auth should
    keep using verify_bearer_token; this is only for "works either way,
    behaves differently if signed in" routes.
    """
    try:
        token = _extract_token(request)
        payload = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
        if await _is_revoked(payload["sub"], payload.get("iat")):
            request.state.user = None
            return None
    except (HTTPException, jwt.PyJWTError):
        request.state.user = None
        return None

    user = {"id": payload["sub"], "email": payload.get("email")}
    request.state.user = user
    request.state.access_token = token
    return user
