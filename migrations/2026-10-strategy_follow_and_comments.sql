-- STS-4/5/6 (scoped subset): leaderboard needs no new storage (computed on
-- demand from published_strategy_forward_snapshots, see
-- services/strategy_leaderboard.py); this migration is for STS-5's
-- alerts-only follow and STS-6's comments/questions-only scope. Ratings
-- (STS-6) and auto-paper-follow (STS-5's other half) are deliberately not
-- built here -- ratings need a 30-day-paper-followed gate that needs
-- auto-paper-follow, which doesn't exist yet.

-- STS-5: who follows which published strategy, for the alerts-only half --
-- NOT auto-paper-follow. Mirrors author_follows exactly.
create table if not exists strategy_follows (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  published_strategy_id bigint not null references published_strategies(id) on delete cascade,
  created_at timestamptz not null default now(),
  constraint strategy_follows_unique unique (user_id, published_strategy_id)
);
create index if not exists strategy_follows_user_idx on strategy_follows(user_id);
create index if not exists strategy_follows_strategy_idx on strategy_follows(published_strategy_id);
grant select, insert, delete on strategy_follows to app_user;
grant select, insert, delete on strategy_follows to app_service;

-- STS-5's in-app alert: "a strategy you follow closed a new trade" -- a 9th
-- branch in web/backend/routers/alerts_inbox.py's UNION, mirroring
-- followed_author_alerts exactly, including RLS (this one IS a
-- per-recipient inbox table, unlike strategy_follows above).
create table if not exists strategy_signal_alerts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  published_strategy_id bigint not null references published_strategies(id) on delete cascade,
  ticker text not null,
  trade_summary text not null,
  created_at timestamptz not null default now(),
  seen_at timestamptz
);
create index if not exists strategy_signal_alerts_user_idx on strategy_signal_alerts(user_id, created_at desc);
alter table strategy_signal_alerts enable row level security;
drop policy if exists strategy_signal_alerts_isolation on strategy_signal_alerts;
create policy strategy_signal_alerts_isolation on strategy_signal_alerts for all
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select, update on strategy_signal_alerts to app_user;
grant select, insert, update on strategy_signal_alerts to app_service;

-- STS-6 (comments/questions half only -- ratings deferred, see note above):
-- mirrors posts.group_id exactly. No membership gate on read/write, unlike
-- groups -- a published strategy's comments are open the same way a
-- ticker-page post with no group_id is.
alter table posts add column if not exists published_strategy_id bigint references published_strategies(id) on delete cascade;
create index if not exists posts_published_strategy_idx on posts(published_strategy_id, created_at desc) where published_strategy_id is not null;
