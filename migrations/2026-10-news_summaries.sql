-- Signal explanation option 2: an LLM summary of each non-earnings 8-K, written once from the filing's own text and
-- stored against the news item. Earnings 8-Ks already have a press-release summary (earnings_release_summaries), so
-- they are not stored here. Safe to re-run.

create table if not exists news_item_summaries (
  news_id bigint primary key references news_items(id) on delete cascade,
  summary text not null,
  method text not null,
  created_at timestamptz not null default now()
);

grant select, insert on news_item_summaries to app_service;
