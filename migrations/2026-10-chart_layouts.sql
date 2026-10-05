-- CHT-7: saved chart layouts per user. Stores the layout settings as JSON and reloads them unchanged.
-- Safe to re-run.

create table if not exists chart_layouts (
  id bigint generated always as identity primary key,
  user_id uuid not null references users(id) on delete cascade,
  name text not null,
  layout jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, name)
);

alter table chart_layouts enable row level security;

drop policy if exists chart_layouts_isolation on chart_layouts;
create policy chart_layouts_isolation on chart_layouts
  using (user_id = current_setting('app.user_id', true)::uuid)
  with check (user_id = current_setting('app.user_id', true)::uuid);

grant select, insert, update, delete on chart_layouts to app_user;
