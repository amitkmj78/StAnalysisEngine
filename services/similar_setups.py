"""DIF-7: past setups that look like today's, and what the price did over the next few sessions.

A past day is "similar" when its stored short-term score is within `band` points of the
current score AND its stored regime label matches the current one. For each similar day
the return from that day's close to the close `horizon` sessions later is measured. Days
with no full forward window are left out.

This is a record of what happened, not a forecast. Below MIN_CASES the result carries a
caveat, because a handful of cases cannot show a pattern.
"""

from datetime import date
from statistics import median
from typing import Optional

import pandas as pd

MIN_CASES = 30
HORIZON_SESSIONS = 10
SCORE_BAND_POINTS = 5.0


def find_similar_setups(
    scores: list[dict],
    closes: pd.Series,
    current_score: float,
    current_regime: Optional[str],
    horizon: int = HORIZON_SESSIONS,
    band: float = SCORE_BAND_POINTS,
) -> dict:
    """scores: [{"date": date, "short_score": float|None, "regime": str|None}], oldest or newest first.
    closes: daily closes indexed by date (a DatetimeIndex or date-like index)."""
    index = [pd.Timestamp(ts).date() for ts in closes.index]
    position = {d: i for i, d in enumerate(index)}
    values = closes.to_numpy(dtype=float)

    returns: list[tuple[date, float]] = []
    for row in scores:
        s = row.get("short_score")
        if s is None or abs(s - current_score) > band:
            continue
        if current_regime is not None and row.get("regime") != current_regime:
            continue
        i = position.get(row["date"])
        if i is None or i + horizon >= len(values) or values[i] <= 0:
            continue
        returns.append((row["date"], (values[i + horizon] / values[i] - 1) * 100))

    returns.sort(key=lambda x: x[0])
    n = len(returns)
    pct = [r for _, r in returns]
    result = {
        "current_score": round(current_score, 2),
        "current_regime": current_regime,
        "horizon_sessions": horizon,
        "score_band_points": band,
        "n": n,
        "median_return_pct": round(median(pct), 2) if pct else None,
        "min_return_pct": round(min(pct), 2) if pct else None,
        "max_return_pct": round(max(pct), 2) if pct else None,
        "caveat": None,
        "note": (
            "A record of what the price did after similar past days. It is not a forecast and not a recommendation."
        ),
    }
    if n < MIN_CASES:
        result["caveat"] = (
            f"Only {n} similar past case{'s' if n != 1 else ''}. Under {MIN_CASES} cases is too few to show a pattern, "
            "so read this as a short record, not a guide."
        )
    return result
