import os
from datetime import datetime, timedelta, timezone

import asyncpg
import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr

from services.password_policy_service import is_breached_password, validate_password_policy

from web.backend.admin import ADMIN_EMAIL
from web.backend.app_settings import PASSWORD_POLICY_ENABLED_KEY, get_setting_bool
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
    if await get_setting_bool(PASSWORD_POLICY_ENABLED_KEY, default=True):
        policy_error = validate_password_policy(body.password)
        if policy_error:
            raise HTTPException(422, policy_error)
        if await is_breached_password(body.password):
            raise HTTPException(
                422,
                "That password has appeared in known data breaches — pick a different one. "
                "(Checked via Have I Been Pwned's k-anonymity API; your password itself is never sent.)",
            )
    elif len(body.password) < 8:
        # Policy toggled off by admin — still enforce an absolute floor,
        # not "any password of any length."
        raise HTTPException(422, "Password must be at least 8 characters.")

    password_hash = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    # The admin account bootstraps itself approved (otherwise nobody could
    # ever approve the very first signup); every other signup starts pending
    # and needs an explicit approval from the admin users page before login
    # will issue a session.
    approved = body.email.lower() == ADMIN_EMAIL.lower()

    async with service_conn() as conn:
        try:
            row = await conn.fetchrow(
                "INSERT INTO users (email, password_hash, approved) VALUES ($1, $2, $3) RETURNING id, email, approved",
                body.email.lower(),
                password_hash,
                approved,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(409, "An account with that email already exists.")

    if not row["approved"]:
        # No session issued — signing up does not grant access until an
        # admin approves the account.
        return {"id": str(row["id"]), "email": row["email"], "pending": True}

    token = _issue_token(str(row["id"]), row["email"])
    _set_session_cookie(response, token)
    # Token included in the body (not just the Set-Cookie header) because the
    # only caller is our own Next.js Server Action, calling server-to-server —
    # its fetch() doesn't forward this response's Set-Cookie to the browser,
    # so it re-sets the cookie itself on the response the browser actually gets.
    return {"id": str(row["id"]), "email": row["email"], "token": token, "pending": False}


@router.post("/login")
async def login(body: LoginRequest, response: Response):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, password_hash, approved FROM users WHERE email = $1",
            body.email.lower(),
        )

    if row is None or not bcrypt.checkpw(body.password.encode(), row["password_hash"].encode()):
        raise HTTPException(401, "Incorrect email or password.")

    if not row["approved"]:
        raise HTTPException(403, "Your account is pending admin approval. You'll be able to sign in once it's approved.")

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
