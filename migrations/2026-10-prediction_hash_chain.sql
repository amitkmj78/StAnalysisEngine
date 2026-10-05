-- DIF-5: one row per day with the chained hash of that day's predictions. The chain is append-only:
-- the app role can add days and read them, never change or delete them. Safe to re-run.

create table if not exists prediction_hash_chain (
  day date primary key,
  record_count integer not null,
  content_hash text not null,
  chain_hash text not null,
  created_at timestamptz not null default now()
);

grant select, insert on prediction_hash_chain to app_service;
