-- NFR-5: real, structured sources (title + url) for a drop alert's
-- sentiment_summary, instead of being discarded after the LLM summarized
-- them. Safe to re-run.

alter table portfolio_drop_alerts add column if not exists sentiment_sources jsonb;
