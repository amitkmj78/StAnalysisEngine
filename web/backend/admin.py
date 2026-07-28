from fastapi import Depends, HTTPException

from web.backend.auth import verify_bearer_token

ADMIN_EMAIL = "amitkmj78@gmail.com"


def require_admin(user: dict = Depends(verify_bearer_token)) -> dict:
    if (user.get("email") or "").lower() != ADMIN_EMAIL.lower():
        raise HTTPException(403, "Admin access required")
    return user
