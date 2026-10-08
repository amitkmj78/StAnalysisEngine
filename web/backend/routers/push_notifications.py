"""ALX-3: web push subscribe/unsubscribe endpoints. The actual send
(services/push_notification_service.py) is wired into
services/notification_dispatcher.py::dispatch_alert as a third channel,
not called from here directly.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from services.push_notification_service import VAPID_PUBLIC_KEY, is_configured

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn

router = APIRouter(prefix="/api/v1/push", tags=["push-notifications"])


@router.get("/vapid-public-key")
async def get_vapid_public_key():
    """Public by design -- the browser needs this to create a
    PushSubscription (PushManager.subscribe({applicationServerKey:
    ...})). No auth required, same as any other public config value."""
    return {"public_key": VAPID_PUBLIC_KEY, "configured": is_configured()}


class SubscriptionKeys(BaseModel):
    p256dh: str
    auth: str


class SubscriptionRequest(BaseModel):
    endpoint: str
    keys: SubscriptionKeys


@router.post("/subscription", dependencies=[Depends(verify_bearer_token)])
async def create_subscription(request: Request, body: SubscriptionRequest):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        await conn.execute(
            """
            INSERT INTO push_subscriptions (user_id, endpoint, p256dh_key, auth_key)
            VALUES ($1::uuid, $2, $3, $4)
            ON CONFLICT (endpoint) DO UPDATE SET
                user_id = $1::uuid, p256dh_key = $3, auth_key = $4
            """,
            user_id, body.endpoint, body.keys.p256dh, body.keys.auth,
        )
    return {"ok": True}


@router.delete("/subscription", dependencies=[Depends(verify_bearer_token)])
async def delete_subscription(request: Request, endpoint: str = Query(...)):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM push_subscriptions WHERE endpoint = $1 RETURNING id", endpoint
        )
    if row is None:
        raise HTTPException(404, "Subscription not found.")
    return {"ok": True}
