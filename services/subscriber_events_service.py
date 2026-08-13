"""
Horizon 1 (docs/signal-licensing-whitelabel-requirements.md.pdf, RS-5
demand instrumentation + RS-6 audit log) — a single append-only event
log backs both: RS-5's conversion/churn/enquiry tracking and RS-6's
actor/timestamp/action/resource audit trail need the same shape, so one
table (subscriber_events) and one write path avoid duplicating the same
insert-only plumbing published_signals already establishes as this app's
pattern for an immutable record.

Read side (demand report, audit log viewer) lives directly in
web/backend/routers/subscriptions.py, matching how every other admin
report in this app queries service_conn directly rather than adding a
parallel reporting service layer.
"""

import json
from typing import Any, Optional

from web.backend.db import service_conn


async def log_event(
    actor_user_id: Optional[str],
    event_type: str,
    resource: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Append one event. actor_user_id is None for a not-yet-signed-in
    visitor (e.g. an anonymous enquiry) — never raises on a logging
    failure being unable to identify who; that's a legitimate event on
    its own. Best-effort is NOT appropriate here (unlike email) since
    this is the audit trail itself — a swallowed exception would defeat
    RS-6's purpose, so this deliberately lets a DB error propagate.
    """
    async with service_conn() as conn:
        await conn.execute(
            """
            INSERT INTO subscriber_events (actor_user_id, event_type, resource, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            """,
            actor_user_id, event_type, resource, json.dumps(metadata) if metadata is not None else None,
        )
