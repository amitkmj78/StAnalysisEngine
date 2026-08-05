from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from services.email_service import send_welcome_email
from web.backend.admin import ADMIN_EMAIL, require_admin
from web.backend.db import service_conn

router = APIRouter(
    prefix="/api/v1/admin/users",
    tags=["admin-users"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
async def list_users():
    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT id, email, approved, created_at FROM users ORDER BY created_at DESC"
        )
    return [
        {
            "id": str(r["id"]),
            "email": r["email"],
            "approved": r["approved"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


@router.post("/{user_id}/approve")
async def approve_user(user_id: str):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET approved = true WHERE id = $1 RETURNING id, email, approved",
            user_id,
        )
    if row is None:
        raise HTTPException(404, "User not found.")
    # Best-effort — send_welcome_email logs and returns False on failure
    # rather than raising, so a broken mail provider can't block approval.
    await run_in_threadpool(send_welcome_email, row["email"])
    return {"id": str(row["id"]), "email": row["email"], "approved": row["approved"]}


@router.post("/{user_id}/reject")
async def reject_user(user_id: str):
    # Only ever deletes a still-pending signup — approving then later
    # wanting to remove someone's access is a separate action, not "reject".
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "DELETE FROM users WHERE id = $1 AND approved = false RETURNING id",
            user_id,
        )
    if row is None:
        raise HTTPException(404, "No pending user with that id.")
    return {"ok": True}


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    # Revokes an already-approved user's access entirely — distinct from
    # reject (which only ever declines a still-pending signup). Cascades to
    # their trades/portfolio/saved_predictions via the FK on those tables.
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        if row is None:
            raise HTTPException(404, "User not found.")
        if row["email"].lower() == ADMIN_EMAIL.lower():
            raise HTTPException(400, "Cannot delete the admin account.")
        await conn.execute("DELETE FROM users WHERE id = $1", user_id)
    return {"ok": True}
