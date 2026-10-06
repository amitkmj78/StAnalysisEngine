-- CHT-5: chart drawings per user and ticker (trend line, horizontal level, rectangle, Fibonacci retracement, text note).
-- Stored on the server so they follow the user across sessions and devices. Row-level security limits each user to
-- their own rows. Safe to re-run.

create table if not exists chart_drawings (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  ticker text not null,
  kind text not null check (kind in ('trend', 'horizontal', 'rectangle', 'fibonacci', 'text')),
  points jsonb not null,
  text text,
  created_at timestamptz not null default now()
);

create index if not exists chart_drawings_user_ticker_idx on chart_drawings (user_id, ticker);

alter table chart_drawings enable row level security;

drop policy if exists chart_drawings_isolation on chart_drawings;
create policy chart_drawings_isolation on chart_drawings
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update, delete on chart_drawings to app_user;
