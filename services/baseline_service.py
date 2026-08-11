"""
Safe Baseline Price Band engine — docs/safe-baseline-price-band-spec.md.

Pure functions only: no I/O, no network, no DB. Callers (the API router)
fetch adjusted OHLC and hand it to compute_baseline_band. This keeps the
engine importable and testable independently of the web layer (NFR-05) and
deterministic for identical inputs (NFR-04).

Scope of this pass ("core band" — US-01/02/06 only, per the spec's own
scoping decision): the five-level ladder, the empirical/sqrt methods,
recency weighting, and the calibration diagnostics. Deliberately NOT
implemented here: the `sum` stress-rail method (FR-13 — internal/research
use only, never user-facing per FR-14, so there's no product need for it
yet), the fill-probability curve (US-03), and first-touch barrier-grid
sequencing (US-04's AC-04.3, FR-18-27 — FR-26 keeps that feature-flagged
off until FR-24's out-of-sample validation exists anyway). AC-04.1/04.2's
simpler shape metrics (rr_ratio, upside_first_rate) ARE in scope and
implemented below, since they're cheap byproducts of the same MAE/MFE
windows and don't need the barrier grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZONS = (10, 30, 60, 90)
DEFAULT_HALF_LIFE_DAYS = 126
MIN_INDEPENDENT_WINDOWS = 3  # FR-04a
WINSORIZE_LOWER_PCT = 0.5
WINSORIZE_UPPER_PCT = 99.5
CALIBRATION_WARNING_THRESHOLD = 0.05  # AC-06.3, 5 percentage points


class InsufficientHistoryError(Exception):
    """
    Raised when FR-04 (raw bar count) or FR-04a (independent window count)
    isn't met. Carries the specific shortfall so the API layer can return a
    structured 422 (NFR-07) — this must never be swallowed into a null or a
    zero-filled response.
    """

    def __init__(self, message: str, bars_required: int, bars_available: int):
        super().__init__(message)
        self.bars_required = bars_required
        self.bars_available = bars_available


def _validate_ohlc(df: pd.DataFrame) -> None:
    """FR-03: reject non-positive or non-finite prices."""
    values = df[["Open", "High", "Low", "Close"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError("OHLC data contains non-positive or non-finite prices.")


def _forward_windows(values: np.ndarray, horizon: int) -> np.ndarray:
    """
    For each valid entry index i (0-based, i = 0 .. N-H-1), returns the H
    values at positions i+1 .. i+H — the forward window strictly after the
    entry day, never including the entry bar itself. Shape (N - H, H).
    """
    forward = values[1:]
    return sliding_window_view(forward, horizon)[: len(values) - horizon]


def _entry_prices(close: np.ndarray, horizon: int) -> np.ndarray:
    """Entry price (that day's Close) for each valid entry index."""
    return close[: len(close) - horizon]


def _winsorize(values: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    lo, hi = np.percentile(values, [lower_pct, upper_pct])
    return np.clip(values, lo, hi)


def _recency_weights(n: int, half_life: Optional[float]) -> np.ndarray:
    """FR-15/16: exponential decay by trading-day age from the most recent
    sample; half_life None or <=0 disables weighting (equal weights)."""
    if not half_life or half_life <= 0:
        return np.ones(n)
    age = (n - 1) - np.arange(n)
    return 0.5 ** (age / half_life)


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, pct: float) -> float:
    """Standard weighted percentile via midpoint cumulative-weight interpolation."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= np.sum(w)
    return float(np.interp(pct / 100.0, cum, v))


def compute_baseline_band(
    df: pd.DataFrame,
    horizon_days: int = 30,
    confidence: float = 0.90,
    method: str = "empirical",
    half_life: Optional[float] = DEFAULT_HALF_LIFE_DAYS,
) -> dict:
    """
    df: daily OHLC with a date index, ascending, columns Open/High/Low/Close
    (FR-01), already split/dividend-adjusted by the caller (FR-02).

    The five ladder levels (floor / accumulation_zone_hi / median_path /
    distribution_zone_lo / ceiling) are all derived from the distribution of
    per-entry-day Maximum Adverse/Favorable Excursion (MAE/MFE) across every
    rolling H-day forward window — "if I'd bought on day i, what's the worst
    and best this position did over the following H trading days":

    - floor / ceiling: the (1-confidence) / confidence percentile of the
      MAE / MFE distribution — calibrated so exactly `confidence` of
      historical H-day windows stayed within [floor, ceiling] (Definitions,
      §2). This is what US-06's breach-rate check verifies.
    - accumulation_zone_hi / distribution_zone_lo: a fixed inner quartile
      (25th / 75th percentile of the same distributions) — a "typical"
      pullback/rally depth, independent of the user's chosen confidence,
      the same way a box plot's IQR doesn't depend on whisker length. Both
      are clamped to the median_path anchor (see below) so a strongly
      trending ticker can't invert the ladder's visual order.
    - median_path: the current last price itself (0%), not a forecast —
      the anchor the other four levels are expressed as a distance from.
      DR-03 requires this never be shown as a prediction; anchoring it to
      today's actual price rather than to any modeled "expected" price is
      what keeps that true structurally, not just by disclaimer.

    Raises ValueError on bad input (FR-03) and InsufficientHistoryError when
    FR-04/FR-04a's minimums aren't met.
    """
    if horizon_days not in HORIZONS:
        raise ValueError(f"horizon_days must be one of {HORIZONS}")
    if not (0.5 <= confidence < 1.0):
        raise ValueError("confidence must be in [0.5, 1.0)")
    if method not in ("empirical", "sqrt"):
        raise ValueError("method must be 'empirical' or 'sqrt'")

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    _validate_ohlc(df)

    bars_required = horizon_days + 2
    if len(df) < bars_required:
        raise InsufficientHistoryError(
            f"Need at least {bars_required} bars for a {horizon_days}-day horizon, have {len(df)}.",
            bars_required, len(df),
        )

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    entry_prices = _entry_prices(close, horizon_days)
    forward_lows = _forward_windows(low, horizon_days)
    forward_highs = _forward_windows(high, horizon_days)

    raw_samples = len(entry_prices)
    # FR-23: non-overlapping-block approximation of independent sample size.
    effective_samples = raw_samples // horizon_days

    if effective_samples < MIN_INDEPENDENT_WINDOWS:
        # FR-04a — the minimum that actually supports a confidence statement,
        # stricter than FR-04's bare-survival bar-count floor above.
        raise InsufficientHistoryError(
            f"Need at least {MIN_INDEPENDENT_WINDOWS} independent {horizon_days}-day windows, "
            f"have {effective_samples} (from {len(df)} bars).",
            bars_required=(MIN_INDEPENDENT_WINDOWS * horizon_days) + 2,
            bars_available=len(df),
        )

    mae_log = np.log(forward_lows.min(axis=1) / entry_prices)
    mfe_log = np.log(forward_highs.max(axis=1) / entry_prices)

    # FR-09 (revised): winsorize the aggregated H-day distribution, not the
    # daily excursions feeding it — protects against a handful of outlier
    # windows without clipping the single worst/best day inside a window
    # that legitimately determines its own MAE/MFE.
    mae_log_w = _winsorize(mae_log, WINSORIZE_LOWER_PCT, WINSORIZE_UPPER_PCT)
    mfe_log_w = _winsorize(mfe_log, WINSORIZE_LOWER_PCT, WINSORIZE_UPPER_PCT)

    weights = _recency_weights(raw_samples, half_life)

    if method == "empirical":
        floor_log = _weighted_percentile(mae_log_w, weights, (1 - confidence) * 100)
        accum_log = min(0.0, _weighted_percentile(mae_log_w, weights, 25))
        distrib_log = max(0.0, _weighted_percentile(mfe_log_w, weights, 75))
        ceiling_log = _weighted_percentile(mfe_log_w, weights, confidence * 100)
    else:
        # sqrt (FR-12): daily dip/rise vs prior close (FR-07), not the H-day
        # MAE/MFE windows above, scaled by sqrt(H) under a Brownian-motion
        # assumption on log-returns — the documented fallback for thin history.
        prior_close = close[:-1]
        daily_dip = np.log(low[1:] / prior_close)
        daily_rise = np.log(high[1:] / prior_close)
        daily_dip_w = _winsorize(daily_dip, WINSORIZE_LOWER_PCT, WINSORIZE_UPPER_PCT)
        daily_rise_w = _winsorize(daily_rise, WINSORIZE_LOWER_PCT, WINSORIZE_UPPER_PCT)
        daily_weights = _recency_weights(len(daily_dip), half_life)
        scale = np.sqrt(horizon_days)
        floor_log = _weighted_percentile(daily_dip_w, daily_weights, (1 - confidence) * 100) * scale
        accum_log = min(0.0, _weighted_percentile(daily_dip_w, daily_weights, 25) * scale)
        distrib_log = max(0.0, _weighted_percentile(daily_rise_w, daily_weights, 75) * scale)
        ceiling_log = _weighted_percentile(daily_rise_w, daily_weights, confidence * 100) * scale

    last_price = float(close[-1])

    def to_level(log_pct: float) -> tuple[float, float]:
        price = last_price * np.exp(log_pct)
        pct = (price / last_price - 1) * 100
        return round(float(price), 4), round(float(pct), 3)

    floor_price, floor_pct = to_level(floor_log)
    accum_price, accum_pct = to_level(accum_log)
    distrib_price, distrib_pct = to_level(distrib_log)
    ceiling_price, ceiling_pct = to_level(ceiling_log)

    # FR-22: breach diagnostics are always about the empirical MAE
    # distribution regardless of `method` — "did price actually breach the
    # stated floor" is a factual question about realized history, not a
    # function of which method produced the floor number.
    breach_mask = mae_log < floor_log
    breach_rate = float(np.average(breach_mask, weights=weights))
    breach_rate_full = float(np.mean(breach_mask))
    recent_cut = max(1, raw_samples // 4)
    breach_rate_recent = float(np.mean(breach_mask[-recent_cut:]))
    expected_breach = 1 - confidence

    # AC-04.1/04.2 — unwinsorized, since these describe realized shape, not
    # a calibrated protective level.
    median_mfe = float(np.median(mfe_log))
    median_mae = float(np.median(mae_log))
    rr_ratio = round(abs(median_mfe / median_mae), 3) if median_mae != 0 else None
    shape_denom = median_mfe - median_mae
    skew = round((median_mfe + median_mae) / shape_denom, 4) if shape_denom != 0 else 0.0

    low_argmin = np.argmin(forward_lows, axis=1)
    high_argmax = np.argmax(forward_highs, axis=1)
    upside_first_rate = round(float(np.mean(high_argmax <= low_argmin)), 4)

    return {
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "last_price": round(last_price, 4),
        "horizon_days": horizon_days,
        "confidence": confidence,
        "method": method,
        "floor": floor_price,
        "floor_pct": floor_pct,
        "accumulation_zone_hi": accum_price,
        "accumulation_zone_hi_pct": accum_pct,
        "median_path": round(last_price, 4),
        "median_path_pct": 0.0,
        "distribution_zone_lo": distrib_price,
        "distribution_zone_lo_pct": distrib_pct,
        "ceiling": ceiling_price,
        "ceiling_pct": ceiling_pct,
        "rr_ratio": rr_ratio,
        "skew": skew,
        "upside_first_rate": upside_first_rate,
        "samples": raw_samples,
        "effective_samples": effective_samples,
        "breach_rate": round(breach_rate, 4),
        "breach_rate_full": round(breach_rate_full, 4),
        "breach_rate_recent": round(breach_rate_recent, 4),
        "expected_breach": round(expected_breach, 4),
        # AC-06.3
        "calibration_warning": abs(breach_rate - expected_breach) > CALIBRATION_WARNING_THRESHOLD,
    }
