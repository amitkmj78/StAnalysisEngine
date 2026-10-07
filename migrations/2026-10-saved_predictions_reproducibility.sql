-- NFR-4: every saved prediction should store the feature values and model
-- version that produced it, so it can be regenerated later -- previously
-- only the final numbers (next_price, signal, ...) were stored. The model
-- is deterministic (fixed random_state, see services/model_service.py's
-- model_version docstring), so this is genuinely sufficient to regenerate
-- a row, not just best-effort disclosure. Safe to re-run.

alter table saved_predictions add column if not exists feature_values jsonb;
alter table saved_predictions add column if not exists model_version text;
