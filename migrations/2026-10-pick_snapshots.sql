-- Last good strategy picks per fund category, stock universe and count. When the live rankings can't be fetched,
-- the plan page shows these with the date they were saved, instead of an empty list. Safe to re-run.

create table if not exists pick_snapshots (
  key text primary key,
  picks jsonb not null,
  saved_at timestamptz not null default now()
);

grant select, insert, update on pick_snapshots to app_service;
