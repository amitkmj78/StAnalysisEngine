"""FND-4: is stock_scores' confidence_score (services.portfolio_compare_
service.derive_confidence's `100 - flip_count*25` heuristic) actually a
calibrated probability of a hit? This fits an empirical hit rate per raw
score from stock_signal_outcomes' real, matured signals (FND-3's table)
and validates it out-of-sample -- a CHRONOLOGICAL fit/holdout split, not a
random one, since these are overlapping-horizon signals (the same non-
independence caveat services/regime_validation.py already states, for
exactly the same reason).

Bucketed by the exact discrete score, not a continuous range:
derive_confidence's raw score only ever takes one of 5 values (0, 25, 50,
75, 100 -- `100 - flip_count*25`, clamped), so CONFIDENCE_BUCKETS' 50-60/
60-70/... ranges (built for the old pipeline's rank-stability proxy,
still used by compute_calibration) would silently drop every score below
50 -- bucketing by the literal value is the correct generalization here.

Scope, disclosed (see the FND-4 tracker note): this produces the honest
calibration REPORT the acceptance criterion names -- a chart a reader can
actually check, "stated vs actual within +-5 points, n>=30". It does NOT
rewrite the confidence NUMBER shown elsewhere in the app today (the stock
detail page's own short_confidence, DIF-1's markers, etc. all still show
derive_confidence's uncalibrated heuristic) -- wiring every one of those
display call sites to a live-fitted lookup, with a matching UI change at
each one, is larger, separable follow-up work. This round's real,
checkable deliverable is proving -- or disproving -- the heuristic
against outcome history, not silently replacing numbers app-wide without
anyone seeing that a number changed meaning.
"""

from typing import Optional

MIN_SAMPLES_PER_BUCKET = 30
AGREEMENT_THRESHOLD_POINTS = 5.0
DEFAULT_FIT_FRACTION = 0.7


def _hit_rate_pct(rows: list[dict]) -> Optional[float]:
    if not rows:
        return None
    return round(sum(1 for r in rows if r["beat_benchmark"]) / len(rows) * 100, 1)


def fit_calibration_table(rows: list[dict], min_samples: int = MIN_SAMPLES_PER_BUCKET) -> dict[float, dict]:
    """score -> {"hit_rate_pct", "n"}, using every row given -- the table
    meant for actual use, fit on as much data as is available (unlike
    validate_calibration_out_of_sample's deliberately smaller fit half).
    A score with fewer than `min_samples` rows is omitted entirely rather
    than returned with an unreliable number."""
    by_score: dict[float, list[dict]] = {}
    for r in rows:
        if r.get("confidence_score") is None:
            continue
        by_score.setdefault(r["confidence_score"], []).append(r)
    return {
        score: {"hit_rate_pct": _hit_rate_pct(group), "n": len(group)}
        for score, group in by_score.items()
        if len(group) >= min_samples
    }


def validate_calibration_out_of_sample(
    rows: list[dict],
    fit_fraction: float = DEFAULT_FIT_FRACTION,
    min_samples: int = MIN_SAMPLES_PER_BUCKET,
) -> dict:
    """Sorts by target_date and splits chronologically: the earlier
    `fit_fraction` of rows fits a hit-rate-per-score table (via
    fit_calibration_table), the later rows validate it. Reports, per
    score: the fit set's rate, the holdout set's own independently-
    computed actual rate, and whether they agree within
    AGREEMENT_THRESHOLD_POINTS -- only when BOTH halves have at least
    `min_samples` rows for that score; otherwise that half's rate is
    None and `agrees_within_5_points` is None too (insufficient data,
    not a false negative)."""
    usable = [r for r in rows if r.get("confidence_score") is not None]
    if not usable:
        return {
            "fit_set_size": 0, "holdout_set_size": 0,
            "min_samples_per_bucket": min_samples,
            "agreement_threshold_points": AGREEMENT_THRESHOLD_POINTS,
            "buckets": {},
        }

    ordered = sorted(usable, key=lambda r: r["target_date"])
    split_idx = int(len(ordered) * fit_fraction)
    fit_rows, holdout_rows = ordered[:split_idx], ordered[split_idx:]

    fit_table = fit_calibration_table(fit_rows, min_samples=min_samples)

    by_score_holdout: dict[float, list[dict]] = {}
    for r in holdout_rows:
        by_score_holdout.setdefault(r["confidence_score"], []).append(r)

    buckets = {}
    for score in sorted(set(fit_table) | set(by_score_holdout)):
        fit_entry = fit_table.get(score)
        holdout_group = by_score_holdout.get(score, [])
        holdout_rate = _hit_rate_pct(holdout_group) if len(holdout_group) >= min_samples else None
        agrees = (
            abs(fit_entry["hit_rate_pct"] - holdout_rate) <= AGREEMENT_THRESHOLD_POINTS
            if fit_entry is not None and holdout_rate is not None
            else None
        )
        buckets[score] = {
            "fit_hit_rate_pct": fit_entry["hit_rate_pct"] if fit_entry else None,
            "fit_n": fit_entry["n"] if fit_entry else 0,
            "holdout_hit_rate_pct": holdout_rate,
            "holdout_n": len(holdout_group),
            "agrees_within_5_points": agrees,
        }

    return {
        "fit_set_size": len(fit_rows),
        "holdout_set_size": len(holdout_rows),
        "min_samples_per_bucket": min_samples,
        "agreement_threshold_points": AGREEMENT_THRESHOLD_POINTS,
        "buckets": buckets,
    }
