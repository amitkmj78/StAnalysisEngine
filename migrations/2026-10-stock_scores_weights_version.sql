-- NFR-4: ties each stock_scores row to the exact SHORT_TERM_WEIGHTS/
-- LONG_TERM_WEIGHTS that produced it (services/stock_score_service.py::
-- weights_version). factor_detail already stores the raw inputs; this is
-- the model-version piece that was missing. Safe to re-run.

alter table stock_scores add column if not exists weights_version text;
