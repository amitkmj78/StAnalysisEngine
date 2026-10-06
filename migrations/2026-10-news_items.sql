-- Signal explanation step 1: SEC 8-K filings stored as news items. One row per filing (keyed by accession number),
-- so a filing that names several tickers is stored once and mapped to each. Filings carry a filing date but no
-- time, so filed_on is a date; nothing here invents a time of day. Safe to re-run.

create table if not exists news_items (
  id bigserial primary key,
  source text not null,
  source_key text not null unique,
  title text not null,
  url text not null,
  publisher text not null,
  filed_on date not null,
  event_type text not null,
  item_codes text[] not null default '{}',
  fetched_at timestamptz not null default now()
);

create table if not exists news_ticker_map (
  news_id bigint not null references news_items(id) on delete cascade,
  ticker text not null,
  primary key (news_id, ticker)
);

create index if not exists news_ticker_map_ticker_idx on news_ticker_map (ticker);

grant select, insert on news_items to app_service;
grant select, insert on news_ticker_map to app_service;
grant usage, select on sequence news_items_id_seq to app_service;
