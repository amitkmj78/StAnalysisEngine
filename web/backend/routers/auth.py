import os
from datetime import datetime, timedelta, timezone

import asyncpg
import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr

from web.backend.auth import SESSION_COOKIE_NAME, verify_bearer_token
from web.backend.db import service_conn

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SESSION_SECRET = os.environ["SESSION_SECRET"]
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
SESSION_TTL = timedelta(days=7)


def _issue_token(user_id: str, email: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "email": email, "iat": now, "exp": now + SESSION_TTL}
    return jwt.encode(payload, SESSION_SECRET, algorithm="HS256")


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        max_age=int(SESSION_TTL.total_seconds()),
        path="/",
    )


class SignupRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup")
async def signup(body: SignupRequest, response: Response):
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters.")

    password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()

    async with service_conn() as conn:
        try:
            row = await conn.fetchrow(
                "INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email",
                body.email.lower(),
                password_hash,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(409, "An account with that email already exists.")

    token = _issue_token(str(row["id"]), row["email"])
    _set_session_cookie(response, token)
    # Token included in the body (not just the Set-Cookie header) because the
    # only caller is our own Next.js Server Action, calling server-to-server —
    # its fetch() doesn't forward this response's Set-Cookie to the browser,
    # so it re-sets the cookie itself on the response the browser actually gets.
    return {"id": str(row["id"]), "email": row["email"], "token": token}


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, password_hash FROM users WHERE email = $1",
            body.email.lower(),
        )

    if row is None or not bcrypt.checkpw(body.password.encode(), row["password_hash"].encode()):
        raise HTTPException(401, "Incorrect email or password.")

    token = _issue_token(str(row["id"]), row["email"])
    _set_session_cookie(response, token)
    return {"id": str(row["id"]), "email": row["email"], "token": token}


@router.post("/logout")
async def logout(response: Response):
    # Stateless JWT — nothing to invalidate server-side in v1, just drop the cookie.
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
async def me(request: Request):
    user = verify_bearer_token(request)
    return user
