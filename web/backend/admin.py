from fastapi import Depends, HTTPException

from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn

ADMIN_EMAIL = "amitkmj78@gmail.com"


async def require_admin(user: dict = Depends(verify_bearer_token)) -> dict:
    # ADMIN_EMAIL is kept as a permanent fallback alongside the role column
    # (not replaced by it) -- a bad or missing role value can never lock out
    # the original account.
    if (user.get("email") or "").lower() == ADMIN_EMAIL.lower():
        return user
    async with service_conn() as conn:
        role = await conn.fetchval("SELECT role FROM users WHERE id = $1::uuid", user["id"])
    if role != "admin":
        raise HTTPException(403, "Admin access required")
    return user
