-- STS-1/2/3: publish a strategy (versioned, locked snapshots), fork a public
-- one into your own workspace, and track its ongoing forward paper record
-- from its publish date. Safe to re-run.

-- STS-2: set only on a forked copy -- a permanent link to the exact
-- published version it was forked from.
alter table saved_strategies add column if not exists forked_from_published_id bigint;
alter table saved_strategies add column if not exists forked_from_version int;

-- STS-1: a published, versioned, locked snapshot of a saved strategy --
-- public data (no RLS), same posture as posts/community_ideas. version=1
-- when root_published_id is null (this row IS the root); v2+ points at the
-- v1 row's id. "History" is just every row in a lineage, ordered by version
-- -- no separate is_latest/superseded flag to keep in sync.
create table if not exists published_strategies (
  id bigint generated always as identity primary key,
  saved_strategy_id bigint references saved_strategies(id) on delete set null,
  author_user_id uuid not null references users(id) on delete cascade,
  root_published_id bigint references published_strategies(id) on delete cascade,
  version int not null default 1,
  name text not null,
  definition jsonb not null,
  result jsonb not null,
  rules_visibility text not null default 'public' check (rules_visibility in ('public', 'summary_only')),
  rules_summary text,
  published_at timestamptz not null default now(),
  constraint published_strategies_summary_required
    check (rules_visibility <> 'summary_only' or rules_summary is not null)
);
create index if not exists published_strategies_author_idx on published_strategies(author_user_id, published_at desc);
create index if not exists published_strategies_lineage_idx on published_strategies(root_published_id, version desc);
grant select, insert on published_strategies to app_user;
grant select, insert on published_strategies to app_service;

-- Now that published_strategies exists, saved_strategies can reference it.
alter table saved_strategies drop constraint if exists saved_strategies_forked_from_published_id_fkey;
alter table saved_strategies add constraint saved_strategies_forked_from_published_id_fkey
  foreign key (forked_from_published_id) references published_strategies(id) on delete set null;

-- STS-3: one row per published strategy per day -- a replay of the exact
-- same backtest math the builder itself uses, over the real
-- [published_at, today] window. Disclosed as a replay, not a brokered
-- paper account (compare alpaca_paper_accounts/paper_account_equity_
-- snapshots, which track a real linked account) -- never confuse the two.
create table if not exists published_strategy_forward_snapshots (
  id bigint generated always as identity primary key,
  published_strategy_id bigint not null references published_strategies(id) on delete cascade,
  as_of_date date not null,
  cumulative_return_pct double precision not null,
  trades int not null default 0,
  created_at timestamptz not null default now(),
  unique (published_strategy_id, as_of_date)
);
-- No separate index needed: the unique constraint above already indexes
-- (published_strategy_id, as_of_date), the only lookup this table needs.
grant select on published_strategy_forward_snapshots to app_user;
grant select, insert on published_strategy_forward_snapshots to app_service;
