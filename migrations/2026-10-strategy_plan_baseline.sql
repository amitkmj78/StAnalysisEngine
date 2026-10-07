-- Saved goals snapshot the portfolio's value when they are saved, so progress starts from zero on day one.
-- Older goals have no snapshot (NULL) and keep the previous calculation. Safe to re-run.

alter table strategy_plans add column if not exists baseline_value double precision;
