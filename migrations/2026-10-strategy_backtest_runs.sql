-- STB-4: one row per strategy backtest run. Counts how many rule variants a user has tried,
-- so the overfitting warning can say so. Insert-only for the app role.
--
-- Deploys skip the schema step after first-time setup (aws_deploy.py), so tables added later
-- must be applied by hand. Run once on the production database, as the postgres superuser:
--   sudo -u postgres psql -d stanalysisengine -f migrations/2026-10-strategy_backtest_runs.sql
-- Safe to re-run: every statement is idempotent.

create table if not exists strategy_backtest_runs (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  definition_hash text not null,
  created_at timestamptz not null default now()
);

create index if not exists strategy_backtest_runs_user_idx on strategy_backtest_runs(user_id, created_at);

alter table strategy_backtest_runs enable row level security;

drop policy if exists strategy_backtest_runs_isolation on strategy_backtest_runs;
create policy strategy_backtest_runs_isolation on strategy_backtest_runs
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert on strategy_backtest_runs to app_user;
