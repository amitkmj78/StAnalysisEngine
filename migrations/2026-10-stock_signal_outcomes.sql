-- FND-3: the public track record's real-signal counterpart to
-- signal_outcomes/published_signals (services/signal_publication_service.py),
-- which is a top-N trailing-return momentum ranking, NOT the stock-page
-- SCR-1/SCR-2 Buy/Hold/Trim signal every ticker actually gets. One row per
-- (ticker, as_of_date, horizon_days) -- every ticker, every day a Buy or
-- Trim signal was issued and has since matured, not just a top-N subset.
-- regime is deliberately NOT stored here -- joined at query time against
-- market_regime_daily, same as the old pipeline already does, so there's
-- no risk of it going stale relative to that table's own backfills.
create table if not exists stock_signal_outcomes (
  id bigint generated always as identity primary key,
  ticker text not null,
  as_of_date date not null,
  horizon_days int not null,
  signal text not null,
  confidence_score real,
  confidence_label text,
  weights_version text,
  entry_price double precision not null,
  exit_price double precision not null,
  realized_return_pct double precision not null,
  benchmark_return_pct double precision not null,
  beat_benchmark boolean not null,
  computed_at timestamptz not null default now(),
  unique (ticker, as_of_date, horizon_days)
);
create index if not exists stock_signal_outcomes_date_idx on stock_signal_outcomes(as_of_date);

grant select on stock_signal_outcomes to app_user;
grant select, insert on stock_signal_outcomes to app_service;
