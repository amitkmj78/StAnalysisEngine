"""
Market Direction Score — Internals pillar (Phase 1 of
docs/market-direction-sentiment-requirements.md). Pure functions only: no
network, no DB. services/market_data_service.py fetches the raw daily
series (breadth, VIX, sector/ratio data); this module turns them into a
score, a regime label, and — for the release gate in spec §9 — forward-
return backtest statistics.

NOT WIRED INTO THE LIVE APP. The spec's own release gate (§9, V-2/V-3)
requires the score to demonstrably improve outcomes before it ships.
Backtested on 5 years of real data (see spec §9 "Validation Results"):
the score as specified here is a statistically significant *contrarian*
signal (worse internals readings preceded better forward SPY returns at
every tested horizon) — the opposite of the "Risk-On -> add risk,
Risk-Off -> get defensive" framing the regime labels imply. Kept here,
tested and unwired, as a validated-negative research artifact rather
than deleted outright — reworking the signal (or just the labels) is a
plausible follow-up, but this must not be surfaced to users, scheduled,
or exposed via any endpoint until it passes the gate.

Deliberately excludes the News and Earnings pillars (deferred to P2/P3
per the spec's own phasing). MDS in this phase is the Internals score
directly; compute_composite_score is written to accept the other two
pillars once they exist, redistributing weight away from whichever
pillars are absent (same graceful-degradation idea as the spec's earnings
off-season coverage rule, generalized to any missing pillar).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

ZSCORE_WINDOW = 250  # ~1 trading year, per spec's "trailing 250-day distribution"
ZSCORE_CLIP = 3.0  # spec: "clipped at ±3σ"
SCALE = 100.0 / ZSCORE_CLIP  # maps a clipped z-score to -100..+100

DEFAULT_PILLAR_WEIGHTS = {"news": 0.25, "earnings": 0.25, "internals": 0.50}  # spec SR-1

REGIME_BANDS = [
    (60, 100, "Risk-On"),
    (20, 60, "Constructive"),
    (-20, 20, "Neutral"),
    (-60, -20, "Cautious"),
    (-100, -60, "Risk-Off"),
]

HYSTERESIS_SESSIONS = 2  # spec SR-5


def _rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score against a trailing window, clipped to ±ZSCORE_CLIP. First
    `window` rows are NaN — there isn't yet a trailing distribution to
    score against, and a z-score computed on a short/empty window would be
    a guess dressed up as a number."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-ZSCORE_CLIP, ZSCORE_CLIP)


def compute_internals_score(internals: pd.DataFrame) -> pd.Series:
    """
    internals: daily DataFrame indexed by date with columns:
      breadth_50dma (0-100, % of universe above 50-DMA),
      vix (level), vix3m (level),
      xly_xlp, hyg_ief, rsp_spy (ratio levels, not returns).

    Five equally-weighted, z-scored sub-signals, per spec §5.1's Internals
    Pillar inputs — breadth level, breadth 5-day change, VIX (inverted:
    high VIX is risk-off), term-structure slope (inverted: an inverted
    curve, VIX > VIX3M, is a stress signal), and risk-appetite momentum
    (21-day % change in the three ratios, averaged, then z-scored):

    Returns a Series of internals scores, -100..+100, NaN wherever the
    trailing 250-day z-score window isn't yet full.
    """
    breadth_level_z = _rolling_zscore(internals["breadth_50dma"])
    breadth_chg_z = _rolling_zscore(internals["breadth_50dma"].diff(5))
    vix_z = -_rolling_zscore(internals["vix"])
    term_slope = internals["vix"] - internals["vix3m"]
    term_slope_z = -_rolling_zscore(term_slope)

    ratio_momentum = pd.concat(
        [
            internals["xly_xlp"].pct_change(21),
            internals["hyg_ief"].pct_change(21),
            internals["rsp_spy"].pct_change(21),
        ],
        axis=1,
    ).mean(axis=1)
    risk_appetite_z = _rolling_zscore(ratio_momentum)

    combined_z = pd.concat(
        [breadth_level_z, breadth_chg_z, vix_z, term_slope_z, risk_appetite_z], axis=1
    ).mean(axis=1)
    return (combined_z * SCALE).clip(-100, 100).rename("internals_score")


def compute_composite_score(
    internals_score: Optional[float],
    news_score: Optional[float] = None,
    earnings_score: Optional[float] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Blends whichever pillars are actually available, redistributing the
    missing pillars' weight proportionally rather than treating a missing
    pillar as zero (which would silently drag the score toward Neutral).
    Returns {"mds": float | None, "data_completeness": float,
    "conflict_flag": bool}. data_completeness = fraction of the three
    pillars present (1.0 in P1 once internals alone is required to be
    present; less once news/earnings exist and one drops out).
    """
    weights = weights or DEFAULT_PILLAR_WEIGHTS
    pillars = {"news": news_score, "earnings": earnings_score, "internals": internals_score}
    present = {k: v for k, v in pillars.items() if v is not None}

    if not present:
        return {"mds": None, "data_completeness": 0.0, "conflict_flag": False}

    present_weight = sum(weights[k] for k in present)
    mds = sum(v * weights[k] for k, v in present.items()) / present_weight

    conflict_flag = False
    values = list(present.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) > 60:  # spec SR-4
                conflict_flag = True

    return {
        "mds": round(float(mds), 2),
        "data_completeness": round(len(present) / len(pillars), 4),
        "conflict_flag": conflict_flag,
    }


def map_regime(mds: Optional[float]) -> Optional[str]:
    if mds is None:
        return None
    for lo, hi, label in REGIME_BANDS:
        if lo <= mds <= hi:
            return label
    return None  # unreachable for a value already clipped to [-100, 100]


def apply_hysteresis(regimes: pd.Series, sessions: int = HYSTERESIS_SESSIONS) -> pd.Series:
    """
    Spec SR-5: a regime change only takes effect once the new band has
    held for `sessions` consecutive days — otherwise a single noisy
    session flips the displayed regime back and forth. Returns the
    "confirmed" regime series (same index), forward-filling the last
    confirmed regime until a new one holds long enough.
    """
    confirmed = []
    current = None
    candidate = None
    candidate_streak = 0

    for regime in regimes:
        if regime == candidate:
            candidate_streak += 1
        else:
            candidate = regime
            candidate_streak = 1

        if candidate_streak >= sessions:
            current = candidate

        confirmed.append(current if current is not None else regime)

    return pd.Series(confirmed, index=regimes.index, name="regime_confirmed")


def run_forward_return_backtest(
    scores: pd.Series,
    price: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 21),
) -> dict:
    """
    Spec V-2: forward returns on `price` (e.g. SPY close) at each horizon,
    conditioned on the regime implied by `scores` on the day the position
    would have been taken. Reports mean, median, hit rate, and t-stat per
    regime per horizon — including for poorly-populated or unflattering
    buckets, per the spec's explicit instruction not to hide those.

    scores and price must share the same index (trading dates). Regime is
    computed from the *unsmoothed* score, matching spec SR-3 ("the raw
    daily value is also persisted... what the backtest consumes").
    """
    regimes = scores.map(map_regime)
    results: dict[int, dict[str, dict]] = {}

    for h in horizons:
        fwd_return = (price.shift(-h) / price - 1.0) * 100.0
        df = pd.DataFrame({"regime": regimes, "fwd_return": fwd_return}).dropna()

        horizon_results = {}
        for _, _, label in REGIME_BANDS:
            bucket = df.loc[df["regime"] == label, "fwd_return"]
            if len(bucket) == 0:
                horizon_results[label] = {
                    "n": 0, "mean_pct": None, "median_pct": None,
                    "hit_rate": None, "t_stat": None, "p_value": None,
                }
                continue
            t_stat, p_value = (
                stats.ttest_1samp(bucket, 0.0) if len(bucket) > 1 else (None, None)
            )
            horizon_results[label] = {
                "n": int(len(bucket)),
                "mean_pct": round(float(bucket.mean()), 3),
                "median_pct": round(float(bucket.median()), 3),
                "hit_rate": round(float((bucket > 0).mean()), 4),
                "t_stat": round(float(t_stat), 3) if t_stat is not None else None,
                "p_value": round(float(p_value), 4) if p_value is not None else None,
            }
        results[h] = horizon_results

    return results
