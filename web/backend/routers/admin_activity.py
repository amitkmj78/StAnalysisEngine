from fastapi import APIRouter, Depends, Query

from web.backend.admin import require_admin
from web.backend.db import service_conn

router = APIRouter(
    prefix="/api/v1/admin/activity",
    tags=["admin-activity"],
    dependencies=[Depends(require_admin)],
)


@router.get("")
async def list_activity(limit: int = Query(200, ge=1, le=1000)):
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT r.id, u.email, r.endpoint, r.created_at
            FROM request_log r
            JOIN users u ON u.id = r.user_id
            ORDER BY r.created_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [
        {
            "id": r["id"],
            "email": r["email"],
            "endpoint": r["endpoint"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]
