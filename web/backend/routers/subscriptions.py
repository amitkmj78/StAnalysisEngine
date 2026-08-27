import csv
import io
import json
import logging
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, EmailStr, Field

from services.email_service import APP_URL
from services.signal_publication_service import DEFAULT_LOOKBACK_DAYS, DEFAULT_UNIVERSE
from services.stripe_service import (
    StripeNotConfiguredError,
    create_checkout_session,
    create_portal_session,
    verify_and_parse_webhook,
)
from services.subscriber_events_service import log_event
from services.subscription_access_service import is_active_paid_subscriber
from web.backend.admin import require_admin
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn
from web.backend.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/subscriptions", tags=["subscriptions"])


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


@router.get("/me", dependencies=[Depends(verify_bearer_token)])
async def get_my_subscription(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT tier, status, current_period_end, created_at, canceled_at
            FROM subscriptions WHERE user_id = $1::uuid
            ORDER BY created_at DESC LIMIT 1
            """,
            user_id,
        )
    if row is None:
        return {"tier": "free", "status": None, "current_period_end": None}
    return {
        "tier": row["tier"],
        "status": row["status"],
        "current_period_end": row["current_period_end"].isoformat() if row["current_period_end"] else None,
        "created_at": row["created_at"].isoformat(),
        "canceled_at": row["canceled_at"].isoformat() if row["canceled_at"] else None,
    }


@router.post("/checkout", dependencies=[Depends(verify_bearer_token)])
async def checkout(request: Request):
    """
    Starts a Stripe Checkout session for the signed-in user. Doesn't
    write a `subscriptions` row itself — that only happens once the
    webhook confirms `checkout.session.completed`, so a browser tab
    closed mid-checkout never leaves a half-created row behind.
    """
    user_id = request.state.user["id"]
    email = request.state.user["email"]
    try:
        url = create_checkout_session(
            user_id=user_id,
            email=email,
            success_url=f"{APP_URL}/subscribe?checkout=success",
            cancel_url=f"{APP_URL}/subscribe?checkout=cancel",
        )
    except StripeNotConfiguredError as e:
        raise HTTPException(503, str(e))
    await log_event(user_id, "checkout_started", resource="subscription")
    return {"url": url}


@router.post("/portal", dependencies=[Depends(verify_bearer_token)])
async def portal(request: Request):
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        stripe_customer_id = await conn.fetchval(
            """
            SELECT stripe_customer_id FROM subscriptions
            WHERE user_id = $1::uuid AND stripe_customer_id IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
            """,
            user_id,
        )
    if stripe_customer_id is None:
        raise HTTPException(404, "No subscription found for this account yet — nothing to manage.")
    try:
        url = create_portal_session(stripe_customer_id, return_url=f"{APP_URL}/subscribe")
    except StripeNotConfiguredError as e:
        raise HTTPException(503, str(e))
    return {"url": url}


@router.post("/webhook")
async def webhook(request: Request):
    """
    Stripe calls this directly — authenticated by webhook signature
    (STRIPE_WEBHOOK_SECRET), not a user session or admin token. One
    subscriptions row per Stripe subscription lifecycle, keyed on
    stripe_subscription_id (unique) rather than user_id, so a
    cancel-then-resubscribe cycle keeps real history for RS-5's cohort
    retention instead of overwriting it. user_id is read from the
    subscription object's own metadata (set at checkout creation via
    subscription_data.metadata), not from event ordering assumptions —
    Stripe doesn't guarantee delivery order between
    checkout.session.completed and customer.subscription.updated.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = verify_and_parse_webhook(payload, sig_header)
    except StripeNotConfiguredError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(400, f"Invalid webhook: {e}")

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("user_id")
        stripe_subscription_id = obj.get("subscription")
        stripe_customer_id = obj.get("customer")
        if user_id and stripe_subscription_id:
            async with service_conn() as conn:
                await conn.execute(
                    """
                    INSERT INTO subscriptions (user_id, tier, status, stripe_customer_id, stripe_subscription_id)
                    VALUES ($1::uuid, 'paid', 'active', $2, $3)
                    ON CONFLICT (stripe_subscription_id) DO NOTHING
                    """,
                    user_id, stripe_customer_id, stripe_subscription_id,
                )
            await log_event(user_id, "checkout_completed", resource=f"subscription:{stripe_subscription_id}")

    elif event_type in ("customer.subscription.updated", "customer.subscription.created"):
        user_id = (obj.get("metadata") or {}).get("user_id")
        stripe_subscription_id = obj.get("id")
        stripe_customer_id = obj.get("customer")
        status = obj.get("status", "active")
        period_end_ts = obj.get("current_period_end")
        period_end = datetime.fromtimestamp(period_end_ts, tz=timezone.utc) if period_end_ts else None
        mapped_status = "active" if status == "active" else ("past_due" if status == "past_due" else "incomplete")
        if user_id and stripe_subscription_id:
            async with service_conn() as conn:
                await conn.execute(
                    """
                    INSERT INTO subscriptions (user_id, tier, status, stripe_customer_id, stripe_subscription_id, current_period_end)
                    VALUES ($1::uuid, 'paid', $2, $3, $4, $5)
                    ON CONFLICT (stripe_subscription_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        current_period_end = EXCLUDED.current_period_end,
                        stripe_customer_id = EXCLUDED.stripe_customer_id
                    """,
                    user_id, mapped_status, stripe_customer_id, stripe_subscription_id, period_end,
                )

    elif event_type == "customer.subscription.deleted":
        user_id = (obj.get("metadata") or {}).get("user_id")
        stripe_subscription_id = obj.get("id")
        stripe_customer_id = obj.get("customer")
        if user_id and stripe_subscription_id:
            async with service_conn() as conn:
                await conn.execute(
                    """
                    INSERT INTO subscriptions (user_id, tier, status, stripe_customer_id, stripe_subscription_id, canceled_at)
                    VALUES ($1::uuid, 'free', 'canceled', $2, $3, now())
                    ON CONFLICT (stripe_subscription_id) DO UPDATE SET
                        tier = 'free', status = 'canceled', canceled_at = now()
                    """,
                    user_id, stripe_customer_id, stripe_subscription_id,
                )
            await log_event(user_id, "subscription_canceled", resource=f"subscription:{stripe_subscription_id}")

    else:
        logger.info("Stripe webhook: unhandled event type %s", event_type)

    return {"received": True}


class EnquiryRequest(BaseModel):
    enquiry_type: str = Field(pattern="^(licensing|api|institutional|other)$")
    contact_email: EmailStr
    message: str = Field(max_length=4000, default="")


@router.post("/enquiry")
@limiter.limit("5/minute")
async def submit_enquiry(request: Request, body: EnquiryRequest):
    """RS-5: the Horizon 2 demand signal — public and unauthenticated by
    design (a prospective licensing/API/institutional customer may not
    have (or want) an account)."""
    user = getattr(request.state, "user", None)
    user_id = user["id"] if user else None
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO demand_enquiries (user_id, enquiry_type, message, contact_email)
            VALUES ($1::uuid, $2, $3, $4)
            """,
            user_id, body.enquiry_type, body.message, body.contact_email,
        )
    await log_event(user_id, "enquiry_submitted", resource=body.enquiry_type, metadata={"contact_email": body.contact_email})
    return {"ok": True}


@router.get("/export/csv", dependencies=[Depends(verify_bearer_token)])
async def export_csv(
    request: Request,
    universe_id: str = Query(DEFAULT_UNIVERSE),
    lookback_days: int = Query(DEFAULT_LOOKBACK_DAYS),
):
    """RS-3: full published-signals history as CSV — paid subscribers only."""
    user_id = request.state.user["id"]
    if not await is_active_paid_subscriber(user_id):
        raise HTTPException(402, "An active paid subscription is required for CSV export.")

    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT target_date, rank, ticker, trailing_return_pct, published_at_utc
            FROM published_signals
            WHERE universe_id = $1 AND lookback_days = $2 AND reason_code IS NULL
            ORDER BY target_date ASC, rank ASC
            """,
            universe_id, lookback_days,
        )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["target_date", "rank", "ticker", "trailing_return_pct", "published_at_utc"])
    for r in rows:
        writer.writerow([r["target_date"], r["rank"], r["ticker"], r["trailing_return_pct"], r["published_at_utc"]])

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=signals_{universe_id}_{lookback_days}d.csv"},
    )


@router.get("/demand-report", dependencies=[Depends(require_admin)])
async def demand_report():
    """RS-5: conversion counts, churn, and cohort retention — computed
    from subscriptions.created_at/canceled_at directly, no separate
    tracking table needed."""
    async with service_conn() as conn:
        totals = await conn.fetchrow(
            """
            SELECT
                count(*) FILTER (WHERE tier = 'paid') AS ever_paid,
                count(*) FILTER (WHERE tier = 'paid' AND status = 'active') AS currently_active,
                count(*) FILTER (WHERE status = 'canceled') AS canceled_total
            FROM subscriptions
            """
        )
        enquiries = await conn.fetch(
            "SELECT enquiry_type, count(*) AS n FROM demand_enquiries GROUP BY enquiry_type ORDER BY n DESC"
        )
        checkout_events = await conn.fetchrow(
            """
            SELECT
                count(*) FILTER (WHERE event_type = 'checkout_started') AS checkout_started,
                count(*) FILTER (WHERE event_type = 'checkout_completed') AS checkout_completed
            FROM subscriber_events
            """
        )

        retention_rows = []
        for months, label in [(1, "1_month"), (3, "3_month"), (6, "6_month")]:
            row = await conn.fetchrow(
                """
                SELECT
                    count(*) AS cohort_size,
                    count(*) FILTER (WHERE status = 'active' OR canceled_at > created_at + ($1 || ' months')::interval) AS retained
                FROM subscriptions
                WHERE tier = 'paid' AND created_at <= now() - ($1 || ' months')::interval
                """,
                str(months),
            )
            cohort_size = row["cohort_size"] or 0
            retained = row["retained"] or 0
            retention_rows.append(
                {
                    "window": label,
                    "cohort_size": cohort_size,
                    "retained": retained,
                    "retention_rate": round(retained / cohort_size, 4) if cohort_size else None,
                }
            )

    monthly_churn = None
    if totals["ever_paid"]:
        one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
        async with service_conn() as conn:
            canceled_last_30d = await conn.fetchval(
                "SELECT count(*) FROM subscriptions WHERE status = 'canceled' AND canceled_at >= $1",
                one_month_ago,
            )
        monthly_churn = round(canceled_last_30d / totals["ever_paid"], 4)

    return {
        "ever_paid_subscribers": totals["ever_paid"],
        "currently_active_subscribers": totals["currently_active"],
        "canceled_total": totals["canceled_total"],
        "monthly_churn_rate": monthly_churn,
        "checkout_started": checkout_events["checkout_started"],
        "checkout_completed": checkout_events["checkout_completed"],
        "checkout_conversion_rate": (
            round(checkout_events["checkout_completed"] / checkout_events["checkout_started"], 4)
            if checkout_events["checkout_started"]
            else None
        ),
        "cohort_retention": retention_rows,
        "enquiries_by_type": [{"enquiry_type": r["enquiry_type"], "count": r["n"]} for r in enquiries],
    }


@router.get("/audit-log", dependencies=[Depends(require_admin)])
async def audit_log(
    event_type: str | None = Query(None),
    actor_user_id: str | None = Query(None),
    since: date | None = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
):
    """RS-6: paginated, filterable read over the same append-only
    subscriber_events table demand_report reads — actor/timestamp/
    action/resource, exactly RS-6's required shape."""
    conditions = []
    params: list = []
    if event_type:
        params.append(event_type)
        conditions.append(f"event_type = ${len(params)}")
    if actor_user_id:
        params.append(actor_user_id)
        conditions.append(f"actor_user_id = ${len(params)}::uuid")
    if since:
        params.append(since)
        conditions.append(f"created_at >= ${len(params)}")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    params.append(limit)
    limit_idx = len(params)
    params.append(offset)
    offset_idx = len(params)

    async with service_conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, actor_user_id, event_type, resource, metadata, created_at
            FROM subscriber_events
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
        )
    return {
        "events": [
            {
                "id": r["id"],
                "actor_user_id": str(r["actor_user_id"]) if r["actor_user_id"] else None,
                "event_type": r["event_type"],
                "resource": r["resource"],
                "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
    }
