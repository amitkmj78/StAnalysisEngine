"""ALX-4: accepts TradingView's own alert webhooks and routes them into
the user's trade journal. No signature verification is possible -- unlike
Plaid/Stripe, TradingView alerts can't sign their own request -- so the
secret token embedded in the URL path IS the authentication, same
"reject before any DB write" posture web/backend/plaid_webhook_verify.py
describes, just via a token lookup instead of a JWT signature.
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from services.notification_dispatcher import dispatch_alert
from services.tradingview_webhook_service import TradingViewAlertError, parse_tradingview_alert
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhooks/tradingview", tags=["tradingview-webhook"])

# Per-token, in-process rate limit -- deliberately NOT the shared
# `limiter` (web/backend/rate_limit.py), which keys by remote address
# when there's no authenticated user (there never is one here): several
# different users' TradingView alerts can legitimately arrive from the
# same small pool of TradingView server IPs, so an IP-keyed limit would
# throttle the wrong thing. Valid as a plain in-process dict because
# this API runs as a single uvicorn process, no --workers (same
# assumption services/yfinance_cache.py's shared cache already relies on).
_RATE_LIMIT_WINDOW_SECONDS = 60
_RATE_LIMIT_MAX_PER_WINDOW = 20
_recent_calls: dict[str, list[float]] = {}


def _rate_limited(token: str) -> bool:
    now = time.monotonic()
    calls = [t for t in _recent_calls.get(token, []) if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    calls.append(now)
    _recent_calls[token] = calls
    return len(calls) > _RATE_LIMIT_MAX_PER_WINDOW


@router.post("/{token}")
async def receive_tradingview_alert(request: Request, token: str):
    if _rate_limited(token):
        raise HTTPException(429, "Too many alerts for this webhook token.")

    async with service_conn() as conn:
        user_id = await conn.fetchval(
            "SELECT user_id FROM user_notification_settings WHERE tradingview_webhook_token = $1", token
        )
    # Reject before touching anything else -- an unrecognized token
    # never triggers a parse attempt or a DB write.
    if user_id is None:
        raise HTTPException(404, "Unknown webhook token.")

    raw_body = await request.body()
    try:
        alert = parse_tradingview_alert(raw_body)
    except TradingViewAlertError as e:
        raise HTTPException(422, str(e))

    trade_id = f"{alert['ticker']}_{int(datetime.now(timezone.utc).timestamp())}_{uuid.uuid4().hex[:6]}"
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO trades (
                trade_id, user_id, ticker, direction, strategy_type, created_at,
                context, status, entry_price, entry_date
            ) VALUES ($1, $2::uuid, $3, $4, 'TradingView', now(), $5, 'OPEN', $6, $7)
            """,
            trade_id, str(user_id), alert["ticker"], alert["direction"], alert["message"],
            alert["price"], datetime.now(timezone.utc) if alert["price"] is not None else None,
        )
    logger.info("TradingView alert routed to journal: ticker=%s user=%s", alert["ticker"], user_id)

    # "routed into their portfolio and journal" per ALX-4's acceptance
    # text -- the journal row above IS the record; this is what makes it
    # seen immediately rather than only noticed on the next journal visit.
    await dispatch_alert(
        str(user_id), alert["ticker"], "tradingview_alert",
        f"{alert['ticker']}: TradingView alert received",
        f"Added to your trade journal: {alert['message']}",
        {"direction": alert["direction"], "price": alert["price"], "trade_id": trade_id},
    )
    return {"ok": True, "trade_id": trade_id, "ticker": alert["ticker"]}
