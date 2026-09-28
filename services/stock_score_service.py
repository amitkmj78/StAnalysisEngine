"""
Pure rules-based composite scoring engine for Phase 1's short-term
(10-90d) and long-term (1-3yr) 0-100 scores (docs/stock-analysis-
requirements.html, SCR-1..4 / EXP-1..3). No DB/HTTP/yfinance imports here
-- same dependency-free boundary services/pit_signal_service.py and
services/ranking_utils.py already keep -- every function takes
already-fetched plain dicts/lists, so this is fully testable with
synthetic data (see services/stock_score_capture_service.py for the I/O
layer that feeds these).

These are rules-based composites (weighted sums of percentile-ranked
factors), NOT a trained ML model -- per EXP-1's own acceptance criteria
("SHAP for tree-based models, weight x standardized value for rules-based
scores"), explainability here is weight x value, not SHAP. The existing
/predict GBM (services/model_service.py) is untouched and unrelated.

Per an explicit product decision: earnings-revisions/earnings-surprise
(short-term) and Quality (long-term) factors are omitted, not faked --
no analyst-estimates or margins/ROE/debt data source exists anywhere in
this app. Short-term score = Momentum + Short-term reversal only.
Long-term score = Value + Growth + Low-volatility only.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Re-exported for callers -- portfolio_compare_service.derive_confidence
# is reused verbatim (same {flip_count, days_captured, current_streak_days,
# unstable} -> {label, score} mapping), not reimplemented here.
from services.portfolio_compare_service import derive_confidence  # noqa: F401

BUY_AT = 70.0
TRIM_AT = 30.0

SHORT_TERM_WEIGHTS = {"momentum": 0.6, "reversal": 0.4}
LONG_TERM_WEIGHTS = {"value": 0.4, "growth": 0.35, "low_vol": 0.25}

# Same threshold/streak convention as web/backend/pit_prices.py's
# get_signal_stability_for_ticker, applied to this new score's own
# signal history instead of pit_quant_signal's.
UNSTABLE_FLIP_THRESHOLD = 3


def percentile_rank(values: dict[str, Optional[float]], lower_is_better: bool = False) -> dict[str, Optional[float]]:
    """0-100 percentile rank of each ticker's raw value within the given
    slice (a universe or a sector group). None values pass through as
    None (excluded from the ranking, never treated as zero)."""
    clean = {k: v for k, v in values.items() if v is not None}
    result: dict[str, Optional[float]] = {k: None for k in values}
    if not clean:
        return result
    series = pd.Series(clean)
    ranked = series.rank(pct=True, ascending=not lower_is_better) * 100.0
    result.update(ranked.to_dict())
    return result


def _weighted_composite(factor_percentiles: dict[str, dict[str, Optional[float]]], weights: dict[str, float]) -> dict[str, Optional[float]]:
    """Weighted sum over each ticker's AVAILABLE factor percentiles,
    re-normalizing weights over just the factors present for that ticker
    -- a ticker missing one factor (e.g. no PIT depth yet for momentum)
    still gets a score built honestly from what IS available, rather than
    None outright, as long as at least one factor exists."""
    tickers: set[str] = set()
    for pct in factor_percentiles.values():
        tickers.update(pct.keys())
    result: dict[str, Optional[float]] = {}
    for ticker in tickers:
        total_weight = 0.0
        weighted_sum = 0.0
        for factor, pct in factor_percentiles.items():
            v = pct.get(ticker)
            if v is None:
                continue
            w = weights[factor]
            weighted_sum += w * v
            total_weight += w
        result[ticker] = round(weighted_sum / total_weight, 2) if total_weight > 0 else None
    return result


def compute_short_score(momentum_pct: dict, reversal_pct: dict) -> dict[str, Optional[float]]:
    return _weighted_composite({"momentum": momentum_pct, "reversal": reversal_pct}, SHORT_TERM_WEIGHTS)


def compute_long_score(value_pct: dict, growth_pct: dict, low_vol_pct: dict) -> dict[str, Optional[float]]:
    return _weighted_composite({"value": value_pct, "growth": growth_pct, "low_vol": low_vol_pct}, LONG_TERM_WEIGHTS)


def sector_percentile(scores: dict[str, Optional[float]], sector_of: dict[str, str]) -> dict[str, Optional[float]]:
    """SCR-3: re-ranks the same scores, scoped to each ticker's own sector
    group -- separate from the universe-wide score/percentile itself."""
    by_sector: dict[str, dict[str, float]] = {}
    for ticker, score in scores.items():
        if score is None:
            continue
        sector = sector_of.get(ticker)
        if sector is None:
            continue
        by_sector.setdefault(sector, {})[ticker] = score
    result: dict[str, Optional[float]] = {t: None for t in scores}
    for group in by_sector.values():
        result.update(percentile_rank(group))
    return result


def score_to_signal(score: Optional[float], buy_at: float = BUY_AT, trim_at: float = TRIM_AT) -> str:
    if score is None:
        return "Hold"
    if score >= buy_at:
        return "Buy"
    if score <= trim_at:
        return "Trim"
    return "Hold"


def compute_factor_contributions(raw_values: dict[str, Optional[float]], percentiles: dict[str, Optional[float]], weights: dict[str, float]) -> list[dict]:
    """EXP-1: contribution = weight * (percentile - 50) per factor, so a
    factor at the universe median contributes ~0 and one far above/below
    median pulls the score up/down proportionally to its weight --
    contributions approximately sum to (score - 50) by construction.
    A factor with no percentile (not enough data) is skipped, not
    zeroed. Sorted by contribution descending; callers pick top/bottom
    via select_top_and_bottom_factors."""
    rows = []
    for factor, pct in percentiles.items():
        if pct is None:
            continue
        contribution = round(weights[factor] * (pct - 50.0), 2)
        rows.append({
            "factor": factor,
            "raw_value": raw_values.get(factor),
            "percentile": round(pct, 1),
            "contribution": contribution,
        })
    rows.sort(key=lambda r: (-r["contribution"], r["factor"]))
    return rows


def select_top_and_bottom_factors(contributions: list[dict], top_n: int = 3, bottom_n: int = 2) -> dict:
    """EXP-1's exact rule: top-3 positive drivers + top-2 negative drags,
    same 'largest positive + N most negative, tie-break by name' shape as
    portfolio_compare_service.select_gap_drivers, generalized from
    holdings to factors."""
    positive = sorted((c for c in contributions if c["contribution"] > 0), key=lambda c: (-c["contribution"], c["factor"]))
    negative = sorted((c for c in contributions if c["contribution"] < 0), key=lambda c: (c["contribution"], c["factor"]))
    return {"drivers": positive[:top_n], "drags": negative[:bottom_n]}


def flip_count_from_signal_history(signals_by_date: list[tuple]) -> Optional[dict]:
    """Builds the {flip_count, days_captured, current_streak_days,
    unstable} shape services.portfolio_compare_service.derive_confidence
    expects, from THIS score's own day-over-day Buy/Hold/Trim history --
    deliberately NOT pit_quant_signal (that's a different signal, reused
    only as a stability PROXY for the old momentum-ranking record in
    Stage B, not here). `signals_by_date`: [(date, signal_str), ...], any
    order. Returns None with fewer than 2 captured days, same as
    web/backend/pit_prices.py's get_signal_stability_for_ticker."""
    if len(signals_by_date) < 2:
        return None
    ordered = sorted(signals_by_date, key=lambda r: r[0])
    flip_count = 0
    current_streak = 1
    for prev, cur in zip(ordered, ordered[1:]):
        if cur[1] == prev[1]:
            current_streak += 1
        else:
            flip_count += 1
            current_streak = 1
    return {
        "flip_count": flip_count,
        "days_captured": len(ordered),
        "current_streak_days": current_streak,
        "unstable": flip_count >= UNSTABLE_FLIP_THRESHOLD,
    }


def flag_12week_trend(history: list[tuple], threshold_pts: float = 15.0, weeks: int = 12) -> dict:
    """SCR-4: resamples daily score history to one point per calendar
    week (the last available day that week), keeps the trailing `weeks`
    points, and flags whether the most recent week-over-week change is
    >= threshold_pts in magnitude. `history`: [(date, score), ...], any
    order."""
    if not history:
        return {"weekly_series": [], "flagged": False, "change_pts": None}
    ordered = sorted(history, key=lambda r: r[0])
    df = pd.DataFrame(ordered, columns=["date", "score"])
    df["date"] = pd.to_datetime(df["date"])
    weekly = df.set_index("date")["score"].resample("W").last().dropna().tail(weeks)
    series = [[d.date().isoformat(), round(float(v), 2)] for d, v in weekly.items()]
    if len(weekly) < 2:
        return {"weekly_series": series, "flagged": False, "change_pts": None}
    change = round(float(weekly.iloc[-1] - weekly.iloc[-2]), 2)
    return {"weekly_series": series, "flagged": abs(change) >= threshold_pts, "change_pts": change}


def weekly_change_explanation(today_detail: dict, week_ago_detail: Optional[dict]) -> Optional[dict]:
    """EXP-3: names the single factor whose contribution moved most since
    ~a week ago. Returns None when there's no week-ago snapshot yet --
    an honest 'not enough history', not a fabricated zero-change.
    today_detail/week_ago_detail: {"factor_name": {"contribution": ..., ...}, ...}."""
    if not week_ago_detail:
        return None
    deltas = []
    for factor, today_val in today_detail.items():
        prev_val = week_ago_detail.get(factor)
        if prev_val is None or today_val is None:
            continue
        delta = round(today_val["contribution"] - prev_val["contribution"], 2)
        deltas.append({"factor": factor, "delta_contribution": delta})
    if not deltas:
        return None
    deltas.sort(key=lambda d: (-abs(d["delta_contribution"]), d["factor"]))
    return deltas[0]
