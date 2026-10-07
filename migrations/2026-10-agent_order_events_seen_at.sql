-- AGT-28: lets agent fills/stop-triggers/failed-run events be dismissed from the
-- generic in-app alerts inbox (web/backend/routers/alerts_inbox.py), the same
-- seen_at convention every other source table there already uses. Safe to re-run.

alter table agent_order_events add column if not exists seen_at timestamptz;
