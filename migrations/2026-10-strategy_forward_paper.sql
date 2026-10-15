-- STB-6: a saved strategy run forward in paper mode with its own live
-- track record. Deliberately NOT a row in alpaca_paper_accounts (that
-- table's (user_id, portfolio_id) uniqueness, and every consumer of it --
-- paper_order_sync.py, challenge_service.py, services/agent/runner.py's
-- _load_paper_account -- assumes "this is the user's one real linked
-- sandbox," not a strategy-owned sleeve). Alpaca's paper sandbox is 1:1
-- with a key pair, so one strategy's own independent track record needs
-- its own separate key pair -- disclosed in the UI before linking.
create table if not exists strategy_paper_accounts (
  id bigint generated always as identity primary key,
  saved_strategy_id bigint not null references saved_strategies(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  api_key_id text not null,
  api_secret_key_encrypted bytea not null,
  alpaca_account_id text,
  account_number text,
  status text not null default 'active',
  enabled boolean not null default false,
  kill_engaged boolean not null default false,
  disclosure_accepted_at timestamptz,
  last_run_at timestamptz,
  last_run_status text,
  last_sync_at timestamptz,
  last_sync_error text,
  created_at timestamptz not null default now()
);
-- One paper account per saved strategy -- a strategy's own independent
-- track record needs exactly one dedicated sandbox, never more than one.
create unique index if not exists strategy_paper_accounts_strategy_idx on strategy_paper_accounts(saved_strategy_id);
alter table strategy_paper_accounts enable row level security;
drop policy if exists strategy_paper_accounts_isolation on strategy_paper_accounts;
create policy strategy_paper_accounts_isolation on strategy_paper_accounts
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select, insert, update, delete on strategy_paper_accounts to app_user;
grant select, update on strategy_paper_accounts to app_service;

-- Per-ticker open-position state carried from one day's run to the next
-- (entry price/ATR/peak for trailing-stop math, days held, cooldown).
-- qty = 0 marks "flat but still in cooldown" so cooldown state doesn't
-- need a second table. NEVER portfolio_positions -- that table backs the
-- Holdings/Strategies UI and must never show a strategy's internal
-- paper sleeve as if it were a user holding.
create table if not exists strategy_paper_positions (
  id bigint generated always as identity primary key,
  strategy_paper_account_id bigint not null references strategy_paper_accounts(id) on delete cascade,
  ticker text not null,
  qty real not null default 0,
  entry_price real,
  entry_atr real,
  peak_price real,
  entry_date date,
  days_held int not null default 0,
  cooldown_until_date date,
  updated_at timestamptz not null default now(),
  unique (strategy_paper_account_id, ticker)
);
alter table strategy_paper_positions enable row level security;
drop policy if exists strategy_paper_positions_isolation on strategy_paper_positions;
create policy strategy_paper_positions_isolation on strategy_paper_positions
  using (strategy_paper_account_id in (
    select id from strategy_paper_accounts where user_id = current_setting('app.user_id', true)::uuid
  ));
grant select on strategy_paper_positions to app_user;
grant select, insert, update, delete on strategy_paper_positions to app_service;

-- Mirrors agent_runs exactly: one row per daily job run per strategy
-- paper account, journaled before anything is sent.
create table if not exists strategy_paper_runs (
  id bigint generated always as identity primary key,
  strategy_paper_account_id bigint not null references strategy_paper_accounts(id) on delete cascade,
  saved_strategy_id bigint not null references saved_strategies(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  status text not null check (status in ('started', 'skipped', 'failed', 'completed')),
  equity real,
  reason text,
  created_at timestamptz not null default now()
);
create index if not exists strategy_paper_runs_account_idx on strategy_paper_runs(strategy_paper_account_id, created_at desc);
alter table strategy_paper_runs enable row level security;
drop policy if exists strategy_paper_runs_isolation on strategy_paper_runs;
create policy strategy_paper_runs_isolation on strategy_paper_runs
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select on strategy_paper_runs to app_user;
grant select, insert on strategy_paper_runs to app_service;

-- Mirrors agent_order_events exactly: insert-only, no app role ever gets
-- update/delete.
create table if not exists strategy_paper_order_events (
  id bigint generated always as identity primary key,
  run_id bigint not null references strategy_paper_runs(id),
  strategy_paper_account_id bigint not null references strategy_paper_accounts(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  event_type text not null,
  ticker text,
  side text,
  qty real,
  est_price real,
  est_value real,
  reason text not null,
  alpaca_order_id text,
  client_order_id text,
  detail jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists strategy_paper_order_events_run_idx on strategy_paper_order_events(run_id, created_at);
alter table strategy_paper_order_events enable row level security;
drop policy if exists strategy_paper_order_events_isolation on strategy_paper_order_events;
create policy strategy_paper_order_events_isolation on strategy_paper_order_events
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select on strategy_paper_order_events to app_user;
grant select, insert on strategy_paper_order_events to app_service;

-- Daily equity snapshot per strategy paper account -- mirrors
-- paper_account_equity_snapshots exactly, feeds compute_member_performance
-- (services/challenge_service.py) unmodified for the track-record math.
create table if not exists strategy_paper_equity_snapshots (
  id bigint generated always as identity primary key,
  strategy_paper_account_id bigint not null references strategy_paper_accounts(id) on delete cascade,
  saved_strategy_id bigint not null references saved_strategies(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  as_of_date date not null,
  equity real not null,
  created_at timestamptz not null default now(),
  unique (strategy_paper_account_id, as_of_date)
);
create index if not exists strategy_paper_equity_snapshots_account_idx on strategy_paper_equity_snapshots(strategy_paper_account_id, as_of_date);
alter table strategy_paper_equity_snapshots enable row level security;
drop policy if exists strategy_paper_equity_snapshots_isolation on strategy_paper_equity_snapshots;
create policy strategy_paper_equity_snapshots_isolation on strategy_paper_equity_snapshots
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);
grant select on strategy_paper_equity_snapshots to app_user;
grant select, insert on strategy_paper_equity_snapshots to app_service;

-- STB-6: self-service per-strategy forward-paper-trading -- defaults off,
-- same "a deploy must not itself turn on a new automated-order-submission
-- feature" convention as every toggle in app_settings.py.
insert into app_settings (key, value) values ('strategy_forward_paper_enabled', 'false') on conflict (key) do nothing;
