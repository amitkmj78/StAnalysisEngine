-- AGT-28: lets agent fills/stop-triggers/failed-run events be dismissed from the
-- generic in-app alerts inbox (web/backend/routers/alerts_inbox.py), the same
-- seen_at convention every other source table there already uses. Safe to re-run.

alter table agent_order_events add column if not exists seen_at timestamptz;
-- app_user previously had select-only on this table (grant select on
-- agent_order_events to app_user, in aws_deploy.py) -- alerts_inbox.py's
-- generic dismiss endpoint (UPDATE ... SET seen_at = now()) needs update
-- too. Missed when this column was added; caught live in production.
grant update on agent_order_events to app_user;
