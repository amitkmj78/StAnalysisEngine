"""ALX-3: web push notifications (the standard browser Push API + VAPID)
-- a third delivery channel alongside email (services/email_service.py)
and the outbound ALR-3 webhook (services/webhook_service.py), both of
which this mirrors: best-effort, never raises, skips (and logs) when
not configured, same fail-open posture _send_email already documents
for missing GMAIL_* env vars.

Scope, disclosed: this is WEB push only. "Mobile" push in ALX-3's own
text would mean a native app's APNs/FCM integration -- there is no
mobile app anywhere in this codebase, only this Next.js web app, and
standard Web Push already reaches a phone's browser (or a PWA installed
from one) without needing a native app shell at all.

VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY are a single, app-wide key pair (not
per-user) -- generate once with `vapid --gen` (installed by the
py-vapid dependency pywebpush pulls in) and set as env vars, same
convention every other API credential in this app already uses
(services/email_service.py's GMAIL_*, etc.). VAPID_PUBLIC_KEY is also
handed to the browser (GET /api/v1/push/vapid-public-key) -- it's
public by design, safe to expose; VAPID_PRIVATE_KEY never leaves the
server.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
# A contact the push service can reach if it needs to flag abuse --
# required by the Web Push protocol, not secret.
VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "mailto:support@stanalysisengine.example")


def is_configured() -> bool:
    return bool(VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY)


def send_push(subscription: dict, title: str, body: str, url: Optional[str] = None) -> tuple[bool, bool]:
    """`subscription`: {"endpoint", "keys": {"p256dh", "auth"}} -- the
    exact shape the browser's own PushSubscription.toJSON() produces,
    as stored in push_subscriptions.

    Returns (sent, should_remove). `should_remove` is True when the push
    service reports the subscription itself is gone (410 Gone / 404 Not
    Found -- the user revoked permission, uninstalled, or cleared site
    data) -- the caller should delete that row rather than keep retrying
    a subscription that will never work again. Any other failure
    (missing VAPID config, a network error, a transient 5xx) returns
    (False, False): never raises, same "a broken delivery channel must
    never block the alert pipeline" posture services.webhook_service.
    send_webhook already documents."""
    if not is_configured():
        logger.warning("Push skipped: VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY not configured")
        return False, False

    from pywebpush import WebPushException, webpush

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps({"title": title, "body": body, "url": url}),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims={"sub": VAPID_SUBJECT},
        )
        return True, False
    except WebPushException as e:
        status = getattr(e.response, "status_code", None) if e.response is not None else None
        should_remove = status in (404, 410)
        logger.warning("Push delivery failed (status=%s): %s", status, e)
        return False, should_remove
    except Exception as e:
        logger.warning("Push delivery failed: %s", e)
        return False, False
