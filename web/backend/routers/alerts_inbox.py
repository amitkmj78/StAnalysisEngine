"""
ALR-2: a single in-app inbox across every alert-producing table --
watchlist_alerts (triggered only), portfolio_drop_alerts,
signal_change_alerts, earnings_alert_log, cost_drop_alerts, and (AGT-28)
agent_order_events for the trading agent's own fills, stop triggers, and
failed runs. No new storage: each row here already exists as its own
table's in-app record, this just UNIONs and normalizes them into one feed,
same "no extra work needed for in-app delivery" point services/
notification_dispatcher.py's docstring makes. Relies on user_conn's RLS
scoping per source table (same convention web/backend/routers/watchlist.py
already uses), not an explicit WHERE user_id filter.

AGT-28 note: the agent's drawdown-circuit-breaker trips (event_type
"agent_risk_state" in services/agent/runner.py) are emailed/webhooked via
dispatch_alert but are not journaled as their own agent_order_events row
(only recorded in agent_runs' own risk_state column), so they do not yet
appear here -- a disclosed gap, not an oversight.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts-inbox"], dependencies=[Depends(verify_bearer_token)])

_SOURCE_TABLES = {
    "watchlist", "condition", "portfolio_drop", "signal_change", "earnings", "cost_drop", "agent",
    "followed_author", "social",
}

_INBOX_QUERY = """
SELECT 'watchlist' AS source, id, ticker, condition_type AS alert_type,
       ('crossed ' || threshold::text || (CASE WHEN triggered_price IS NOT NULL
            THEN ' (now ' || triggered_price::text || ')' ELSE '' END)) AS summary,
       created_at, triggered_at AS event_at, seen_at
FROM watchlist_alerts WHERE triggered_at IS NOT NULL

UNION ALL

SELECT 'condition' AS source, id, ticker, 'condition_alert' AS alert_type,
       COALESCE(triggered_detail, '') AS summary,
       created_at, triggered_at AS event_at, seen_at
FROM condition_alerts WHERE triggered_at IS NOT NULL

UNION ALL

SELECT 'portfolio_drop' AS source, id, ticker, 'portfolio_drop' AS alert_type,
       ('down ' || round(pct_change::numeric, 2)::text || '% from prior close ($' || prev_close::text
            || ' -> $' || price_at_check::text || ')') AS summary,
       created_at, created_at AS event_at, seen_at
FROM portfolio_drop_alerts

UNION ALL

SELECT 'signal_change' AS source, id, ticker, 'signal_change' AS alert_type,
       (horizon || '-term signal: ' || COALESCE(old_signal, '?') || ' -> ' || COALESCE(new_signal, '?')) AS summary,
       created_at, created_at AS event_at, seen_at
FROM signal_change_alerts

UNION ALL

SELECT 'earnings' AS source, id, ticker, 'earnings' AS alert_type,
       ('reports earnings ' || earnings_date::text) AS summary,
       created_at, created_at AS event_at, seen_at
FROM earnings_alert_log

UNION ALL

SELECT 'cost_drop' AS source, id, ticker, 'cost_drop' AS alert_type,
       ('down ' || round(pct_change::numeric, 2)::text || '% from cost ($' || avg_cost::text
            || ' -> $' || current_price::text || ')') AS summary,
       created_at, created_at AS event_at, seen_at
FROM cost_drop_alerts

UNION ALL

SELECT 'agent' AS source, id, ticker, COALESCE(trigger, event_type) AS alert_type,
       reason AS summary,
       created_at, created_at AS event_at, seen_at
FROM agent_order_events
WHERE event_type IN ('filled', 'run_failed')

UNION ALL

SELECT 'followed_author' AS source, id, ticker, 'followed_author_idea' AS alert_type,
       ('a new idea on ' || ticker) AS summary,
       created_at, created_at AS event_at, seen_at
FROM followed_author_alerts

UNION ALL

-- SOC-9: new_follower/post_reply/mention/group_activity -- one
-- consolidated table (see migrations/2026-10-social_network.sql)
-- rather than a near-identical branch per sub-type.
SELECT 'social' AS source, sn.id, p.ticker, sn.notification_type AS alert_type,
       sn.summary,
       sn.created_at, sn.created_at AS event_at, sn.seen_at
FROM social_notifications sn LEFT JOIN posts p ON p.id = sn.post_id

ORDER BY event_at DESC
LIMIT 200
"""


def _record_to_dict(record) -> dict:
    d = {k: record[k] for k in record.keys()}
    d["link"] = f"/stock/{d['ticker']}" if d["ticker"] else None
    return d


@router.get("")
async def list_inbox(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        rows = await conn.fetch(_INBOX_QUERY)
    return [_record_to_dict(r) for r in rows]


@router.post("/{source}/{alert_id}/dismiss")
async def dismiss_inbox_item(request: Request, source: str, alert_id: int):
    """Marks one row from any of the source tables as seen -- the
    table is chosen from `source` (validated against a fixed allowlist,
    never interpolated from unchecked user input) rather than exposing
    one near-duplicate dismiss endpoint per table."""
    if source not in _SOURCE_TABLES:
        raise HTTPException(422, f"source must be one of {sorted(_SOURCE_TABLES)}")
    table = {
        "watchlist": "watchlist_alerts",
        "condition": "condition_alerts",
        "portfolio_drop": "portfolio_drop_alerts",
        "signal_change": "signal_change_alerts",
        "earnings": "earnings_alert_log",
        "cost_drop": "cost_drop_alerts",
        "agent": "agent_order_events",
        "followed_author": "followed_author_alerts",
        "social": "social_notifications",
    }[source]

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(f"UPDATE {table} SET seen_at = now() WHERE id = $1 RETURNING id", alert_id)
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}
