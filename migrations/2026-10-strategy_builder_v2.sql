-- Strategy Builder v2: store each run's daily Sharpe (for the deflated Sharpe across a user's variants),
-- and saved strategies with optional read-only share links. Safe to re-run.

alter table strategy_backtest_runs add column if not exists sharpe_daily double precision;

create table if not exists saved_strategies (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null,
  definition jsonb not null,
  result jsonb not null,
  data_end date,
  share_token text unique,
  created_at timestamptz not null default now()
);

create index if not exists saved_strategies_user_idx on saved_strategies(user_id, created_at desc);

alter table saved_strategies enable row level security;

drop policy if exists saved_strategies_isolation on saved_strategies;
create policy saved_strategies_isolation on saved_strategies
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update, delete on saved_strategies to app_user;
grant select, update on saved_strategies to app_service;
