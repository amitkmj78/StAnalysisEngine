-- The home screen (Stock Detail) needs a ticker to open on. NULL means "no
-- user override yet", and the app falls back to SPY. Safe to re-run.

alter table users add column if not exists default_ticker text;
