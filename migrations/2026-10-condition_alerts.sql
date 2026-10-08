-- ALX-1: user-defined multi-condition alerts (price, indicator, score,
-- signal, regime, earnings), combined with a single AND/OR across the
-- whole set -- see services/condition_alert_service.py. Mirrors
-- watchlist_alerts' shape/RLS.
create table if not exists condition_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  conditions jsonb not null,
  combinator text not null,
  active boolean not null default true,
  triggered_at timestamptz,
  triggered_detail text,
  seen_at timestamptz,
  created_at timestamptz not null default now()
);
create index if not exists condition_alerts_user_idx on condition_alerts(user_id, created_at desc);
alter table condition_alerts enable row level security;
drop policy if exists condition_alerts_isolation on condition_alerts;
create policy condition_alerts_isolation on condition_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update, delete on condition_alerts to app_user;
grant select, update on condition_alerts to app_service;
