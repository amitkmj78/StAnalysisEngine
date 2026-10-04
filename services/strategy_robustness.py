"""SB-V2, SB-V4, SB-V5: walk-forward windows, parameter sensitivity, deflated Sharpe.

Walk-forward (SB-V2): the rules are fixed (nothing is refitted), so each test window is simply a
later stretch of dates. The windows are rolled forward through the period; the share of test windows
where the strategy beat the same-stock buy-and-hold is reported. The training part of each window is
not used for fitting, and the result says so.

Sensitivity (SB-V4): each numeric threshold is moved by -10% and +10%, one at a time, and the Sharpe is
recomputed. A large swing means the result depends on a precise threshold, not a general pattern.

Deflated Sharpe (SB-V5): the probability that the strategy's Sharpe is above what the best of N
no-skill trials would reach by chance, using the Sharpe values of the variants this user has actually
tried (Bailey and Lopez de Prado, 2014). Skew and kurtosis are taken from the strategy's own daily
returns.
"""

import math
from statistics import NormalDist
from typing import Callable, Optional

import numpy as np
import pandas as pd

WINDOW_TRAIN_SESSIONS = 504   # about two years
WINDOW_TEST_SESSIONS = 126    # about six months
SENSITIVITY_STEP = 0.10
EULER_GAMMA = 0.5772156649

_N = NormalDist()


def _sharpe_per_period(values: np.ndarray) -> Optional[float]:
    if len(values) < 2:
        return None
    sd = float(values.std(ddof=1))
    if sd == 0:
        return None
    return float(values.mean() / sd)


def walk_forward(strategy_pct: pd.Series, basket_pct: pd.Series,
                 train: int = WINDOW_TRAIN_SESSIONS, test: int = WINDOW_TEST_SESSIONS) -> dict:
    """Rolls test windows of `test` sessions, starting after the first `train` sessions."""
    s = strategy_pct.dropna()
    b = basket_pct.reindex(s.index).dropna()
    s = s.reindex(b.index)
    windows = []
    start = train
    while start + test <= len(s):
        seg = slice(start, start + test)
        strat_total = float(((1 + s.iloc[seg] / 100).prod() - 1) * 100)
        basket_total = float(((1 + b.iloc[seg] / 100).prod() - 1) * 100)
        windows.append({
            "start": str(s.index[start].date()),
            "end": str(s.index[start + test - 1].date()),
            "strategy_pct": round(strat_total, 2),
            "basket_pct": round(basket_total, 2),
            "beat_basket": strat_total > basket_total,
        })
        start += test
    beats = sum(1 for w in windows if w["beat_basket"])
    return {
        "train_sessions": train,
        "test_sessions": test,
        "windows": windows,
        "test_windows": len(windows),
        "beat_basket_windows": beats,
        "beat_basket_pct": round(beats / len(windows) * 100, 1) if windows else None,
        "note": "Rules are fixed, not refitted: the training part of each window is context only. "
                "Each test window is a later stretch of dates.",
    }


def sensitivity(run_sharpe: Callable[[dict], Optional[float]], base: dict[str, float],
                step: float = SENSITIVITY_STEP) -> dict:
    """run_sharpe(params) -> annual Sharpe for those threshold values. base: {param_name: value}.
    Each parameter is moved by -step and +step, one at a time."""
    base_sharpe = run_sharpe(dict(base))
    rows = []
    for name, value in base.items():
        cells = []
        for factor in (1 - step, 1.0, 1 + step):
            params = dict(base)
            params[name] = value * factor
            sh = base_sharpe if factor == 1.0 else run_sharpe(params)
            cells.append({"factor": round(factor, 2), "value": round(params[name], 4),
                          "sharpe": None if sh is None else round(sh, 3)})
        values = [c["sharpe"] for c in cells if c["sharpe"] is not None]
        swing = round(max(values) - min(values), 3) if len(values) >= 2 else None
        rows.append({"parameter": name, "cells": cells, "swing": swing})
    widest = max((r["swing"] for r in rows if r["swing"] is not None), default=None)
    return {
        "step_pct": round(step * 100),
        "base_sharpe": None if base_sharpe is None else round(base_sharpe, 3),
        "rows": rows,
        "widest_swing": widest,
        "note": "Each threshold moved by the step, one at a time. A wide swing means the result depends on that exact value.",
    }


def deflated_sharpe(daily_returns_pct: pd.Series, trial_sharpes_daily: list[float]) -> dict:
    """Probability (0 to 1) that the strategy's Sharpe is genuinely above zero after allowing for the
    number of variants tried. trial_sharpes_daily: the per-period Sharpe of each variant this user ran,
    including this one. With one variant the correction is zero."""
    r = pd.Series(daily_returns_pct, dtype=float).dropna() / 100
    T = len(r)
    sr = _sharpe_per_period(r.to_numpy())
    if sr is None or T < 3:
        return {"probability": None, "variants": len(trial_sharpes_daily), "note": "Not enough data."}
    skew = float(r.skew()) if T > 2 else 0.0
    kurt = float(r.kurt()) + 3.0 if T > 3 else 3.0  # pandas gives excess kurtosis; the formula uses the raw value
    N = max(1, len(trial_sharpes_daily))
    if N > 1:
        var = float(np.var(trial_sharpes_daily, ddof=1)) if len(trial_sharpes_daily) > 1 else 0.0
        sr0 = math.sqrt(var) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / N) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (N * math.e)))
    else:
        sr0 = 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return {"probability": None, "variants": N, "note": "Return distribution too extreme for this approximation."}
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)
    prob = _N.cdf(z)
    return {
        "probability": round(prob, 3),
        "sharpe_daily": round(sr, 4),
        "benchmark_sharpe_daily": round(sr0, 4),
        "variants": N,
        "note": (
            f"Probability that the Sharpe is above zero after allowing for {N} variant(s) tried. "
            "Computed from the Sharpe values of the variants this user has run in the last 90 days."
        ),
    }
