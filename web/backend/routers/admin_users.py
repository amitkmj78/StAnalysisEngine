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
    """Includes each user's portfolio_count/position_count (across all their
    portfolios) so an admin deciding whether to deactivate an account can
    see at a glance whether it's actually in use."""
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT
                u.id, u.email, u.approved, u.is_active, u.created_at,
                count(DISTINCT p.id) AS portfolio_count,
                count(pp.id) AS position_count
            FROM users u
            LEFT JOIN portfolios p ON p.user_id = u.id
            LEFT JOIN portfolio_positions pp ON pp.portfolio_id = p.id AND pp.user_id = u.id
            GROUP BY u.id, u.email, u.approved, u.is_active, u.created_at
            ORDER BY u.created_at DESC
            """
        )
    return [
        {
            "id": str(r["id"]),
            "email": r["email"],
            "approved": r["approved"],
            "is_active": r["is_active"],
            "created_at": r["created_at"].isoformat(),
            "portfolio_count": r["portfolio_count"],
            "position_count": r["position_count"],
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
    sent = await run_in_threadpool(send_welcome_email, row["email"])
    return {
        "id": str(row["id"]),
        "email": row["email"],
        "approved": row["approved"],
        "welcome_email_sent": sent,
    }


@router.post("/{user_id}/send-welcome-email")
async def send_welcome_email_now(user_id: str):
    """Manual (re)send — for a user whose auto-send-on-approve failed (e.g.
    mail wasn't configured yet at the time) or who just wants the info
    again. Unlike the approval flow, a failure here is reported back rather
    than swallowed, since sending the email is the whole point of the
    admin clicking this button."""
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT email, approved FROM users WHERE id = $1", user_id)
    if row is None:
        raise HTTPException(404, "User not found.")
    if not row["approved"]:
        raise HTTPException(400, "User is not approved yet — approve them first.")
    sent = await run_in_threadpool(send_welcome_email, row["email"])
    if not sent:
        raise HTTPException(
            502,
            "Email failed to send — check GMAIL_SENDER_EMAIL/GMAIL_APP_PASSWORD are configured "
            "correctly and check server logs for the underlying error.",
        )
    return {"ok": True, "email": row["email"]}


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


@router.post("/{user_id}/deactivate")
async def deactivate_user(user_id: str):
    # Reversible suspension, distinct from delete: blocks future logins
    # (checked in POST /login) but keeps the account and all its
    # trades/portfolio/saved_predictions intact. An already-open session
    # keeps working until it next logs in — see the is_active comment on
    # the users table migration for why.
    async with service_conn() as conn:
        row = await conn.fetchrow("SELECT email FROM users WHERE id = $1", user_id)
        if row is None:
            raise HTTPException(404, "User not found.")
        if row["email"].lower() == ADMIN_EMAIL.lower():
            raise HTTPException(400, "Cannot deactivate the admin account.")
        row = await conn.fetchrow(
            "UPDATE users SET is_active = false WHERE id = $1 RETURNING id, email, is_active",
            user_id,
        )
    return {"id": str(row["id"]), "email": row["email"], "is_active": row["is_active"]}


@router.post("/{user_id}/reactivate")
async def reactivate_user(user_id: str):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET is_active = true WHERE id = $1 RETURNING id, email, is_active",
            user_id,
        )
    if row is None:
        raise HTTPException(404, "User not found.")
    return {"id": str(row["id"]), "email": row["email"], "is_active": row["is_active"]}


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


@router.get("/{user_id}/portfolios")
async def list_user_portfolios(user_id: str):
    """Per-portfolio detail for one user — the drill-down from the
    portfolio_count shown in GET /admin/users, so an admin can see (and
    act on) individual portfolios rather than only the account as a whole."""
    async with service_conn() as conn:
        user = await conn.fetchrow("SELECT id FROM users WHERE id = $1", user_id)
        if user is None:
            raise HTTPException(404, "User not found.")
        rows = await conn.fetch(
            """
            SELECT p.id, p.name, p.is_active, p.created_at, count(pp.id) AS position_count
            FROM portfolios p
            LEFT JOIN portfolio_positions pp ON pp.portfolio_id = p.id AND pp.user_id = p.user_id
            WHERE p.user_id = $1::uuid
            GROUP BY p.id, p.name, p.is_active, p.created_at
            ORDER BY p.created_at ASC
            """,
            user_id,
        )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "is_active": r["is_active"],
            "created_at": r["created_at"].isoformat(),
            "position_count": r["position_count"],
        }
        for r in rows
    ]


async def _auto_deactivate_if_no_active_portfolios(conn, user_id: str) -> bool:
    """If a user has just lost their last active portfolio (deactivated or
    deleted), also deactivate the account itself — mirroring
    deactivate_user's reversible-suspension behavior, since an account with
    nothing active left isn't meaningfully usable. Skipped for the admin
    account, same guard as the direct deactivate/delete endpoints. Returns
    whether the account was actually flipped."""
    remaining_active = await conn.fetchval(
        "SELECT count(*) FROM portfolios WHERE user_id = $1::uuid AND is_active", user_id
    )
    if remaining_active > 0:
        return False
    user_row = await conn.fetchrow("SELECT email, is_active FROM users WHERE id = $1", user_id)
    if user_row is None or not user_row["is_active"] or user_row["email"].lower() == ADMIN_EMAIL.lower():
        return False
    await conn.execute("UPDATE users SET is_active = false WHERE id = $1", user_id)
    return True


@router.post("/{user_id}/portfolios/{portfolio_id}/deactivate")
async def deactivate_user_portfolio(user_id: str, portfolio_id: int):
    """Reversible, same shape as deactivate_user: hides this one portfolio
    from the owning user (blocked in _resolve_portfolio_id, dropped from
    GET /list) without touching its positions/strategies/alerts. If this
    was the user's last active portfolio, the account is auto-deactivated
    too — see _auto_deactivate_if_no_active_portfolios."""
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET is_active = false WHERE id = $1 AND user_id = $2::uuid RETURNING id, name, is_active",
            portfolio_id, user_id,
        )
        if row is None:
            raise HTTPException(404, "Portfolio not found.")
        user_deactivated = await _auto_deactivate_if_no_active_portfolios(conn, user_id)
    return {
        "id": row["id"],
        "name": row["name"],
        "is_active": row["is_active"],
        "user_auto_deactivated": user_deactivated,
    }


@router.post("/{user_id}/portfolios/{portfolio_id}/reactivate")
async def reactivate_user_portfolio(user_id: str, portfolio_id: int):
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET is_active = true WHERE id = $1 AND user_id = $2::uuid RETURNING id, name, is_active",
            portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"id": row["id"], "name": row["name"], "is_active": row["is_active"]}


@router.delete("/{user_id}/portfolios/{portfolio_id}")
async def delete_user_portfolio(user_id: str, portfolio_id: int):
    """Permanent — distinct from deactivate above. Cascades to this
    portfolio's positions/strategies/alerts via the FK on those tables. If
    this was the user's last active portfolio, the account is auto-
    deactivated too, same as the deactivate endpoint."""
    async with service_conn() as conn:
        row = await conn.fetchrow(
            "DELETE FROM portfolios WHERE id = $1 AND user_id = $2::uuid RETURNING id",
            portfolio_id, user_id,
        )
        if row is None:
            raise HTTPException(404, "Portfolio not found.")
        user_deactivated = await _auto_deactivate_if_no_active_portfolios(conn, user_id)
    return {"ok": True, "user_auto_deactivated": user_deactivated}
