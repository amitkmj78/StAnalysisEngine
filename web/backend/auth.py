import os

import jwt
from fastapi import HTTPException, Request, status

SESSION_SECRET = os.environ["SESSION_SECRET"]
SESSION_COOKIE_NAME = "session"


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


def verify_bearer_token(request: Request) -> dict:
    token = _extract_token(request)

    try:
        payload = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid session: {exc}")

    user = {"id": payload["sub"], "email": payload.get("email")}
    request.state.user = user
    request.state.access_token = token
    return user


def verify_bearer_token_optional(request: Request) -> dict | None:
    """
    Same as verify_bearer_token but never raises — returns None for a
    missing or invalid token instead of a 401. Needed for RS-2's tiered
    content gating: the free tier must stay reachable by a fully
    anonymous visitor, while a signed-in paid subscriber on the same
    endpoint still gets identified. Endpoints that require auth should
    keep using verify_bearer_token; this is only for "works either way,
    behaves differently if signed in" routes.
    """
    try:
        token = _extract_token(request)
        payload = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except (HTTPException, jwt.PyJWTError):
        request.state.user = None
        return None

    user = {"id": payload["sub"], "email": payload.get("email")}
    request.state.user = user
    request.state.access_token = token
    return user
