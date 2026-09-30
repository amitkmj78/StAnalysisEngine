"""
ALR-2: a single in-app inbox across every alert-producing table --
watchlist_alerts (triggered only), portfolio_drop_alerts,
signal_change_alerts, earnings_alert_log, cost_drop_alerts. No new
storage: each row here already exists as its own table's in-app record,
this just UNIONs and normalizes them into one feed, same "no extra work
needed for in-app delivery" point services/notification_dispatcher.py's
docstring makes. Relies on user_conn's RLS scoping per source table
(same convention web/backend/routers/watchlist.py already uses), not an
explicit WHERE user_id filter.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts-inbox"], dependencies=[Depends(verify_bearer_token)])

_SOURCE_TABLES = {"watchlist", "portfolio_drop", "signal_change", "earnings", "cost_drop"}

_INBOX_QUERY = """
SELECT 'watchlist' AS source, id, ticker, condition_type AS alert_type,
       ('crossed ' || threshold::text || (CASE WHEN triggered_price IS NOT NULL
            THEN ' (now ' || triggered_price::text || ')' ELSE '' END)) AS summary,
       created_at, triggered_at AS event_at, seen_at
FROM watchlist_alerts WHERE triggered_at IS NOT NULL

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
    """Marks one row from any of the five source tables as seen -- the
    table is chosen from `source` (validated against a fixed allowlist,
    never interpolated from unchecked user input) rather than exposing
    five near-duplicate dismiss endpoints."""
    if source not in _SOURCE_TABLES:
        raise HTTPException(422, f"source must be one of {sorted(_SOURCE_TABLES)}")
    table = {
        "watchlist": "watchlist_alerts",
        "portfolio_drop": "portfolio_drop_alerts",
        "signal_change": "signal_change_alerts",
        "earnings": "earnings_alert_log",
        "cost_drop": "cost_drop_alerts",
    }[source]

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(f"UPDATE {table} SET seen_at = now() WHERE id = $1 RETURNING id", alert_id)
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}
