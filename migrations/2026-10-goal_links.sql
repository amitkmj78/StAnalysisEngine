-- STRAT-10: holdings linked to a saved goal. A link takes a share of one holding's value (share_pct), and the same
-- shares can't be counted toward two goals (enforced by the API). baseline_value is that share's value when linked,
-- so the goal's progress starts from zero. Row-level security limits each user to their own links. Safe to re-run.

create table if not exists goal_links (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  plan_id bigint not null references strategy_plans(id) on delete cascade,
  portfolio_id bigint not null references portfolios(id) on delete cascade,
  ticker text not null,
  share_pct double precision not null check (share_pct > 0 and share_pct <= 100),
  role text not null check (role in ('core', 'pick')),
  baseline_value double precision not null default 0,
  created_at timestamptz not null default now(),
  unique (plan_id, portfolio_id, ticker)
);

create index if not exists goal_links_user_idx on goal_links (user_id);

alter table goal_links enable row level security;

drop policy if exists goal_links_isolation on goal_links;
create policy goal_links_isolation on goal_links
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update, delete on goal_links to app_user;
