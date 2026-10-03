"""REG-10: validate the regime label on forward risk, not forward return.

Agreed test (see the approved plan; thresholds are not tuned on its output):
  - For every date with a label, the realised volatility and worst drawdown of
    SPY over the next HORIZON_SESSIONS trading days are attached to that date.
  - Pass for the label: Risk-Off and Cautious days show higher average forward
    volatility AND deeper average worst drawdown than Risk-On days.
  - Pass for the divergence flag: flagged days show higher forward volatility
    AND deeper worst drawdown than unflagged days.
  - A group with fewer than MIN_DAYS observations reports 'insufficient data'
    instead of passing or failing.

Forward windows overlap, so consecutive days are not independent. The report
states the number of observations and this caveat; it does not claim
significance.
"""

from typing import Optional

import numpy as np
import pandas as pd

HORIZON_SESSIONS = 21
MIN_DAYS = 30
RISK_OFF_LABELS = ("Cautious", "Risk-Off")


def forward_risk(spy_close: pd.Series, horizon: int = HORIZON_SESSIONS) -> pd.DataFrame:
    """Per date: annualized realised vol of the next `horizon` daily returns, and
    the worst peak-to-trough drawdown over the same window (starting at that date's close)."""
    closes = spy_close.dropna().astype(float)
    returns = closes.pct_change()
    rows = []
    values = closes.values
    for i in range(len(closes) - horizon):
        window_ret = returns.iloc[i + 1: i + 1 + horizon].values
        path = values[i: i + horizon + 1]
        peaks = np.maximum.accumulate(path)
        worst = float(np.min(path / peaks - 1.0)) * 100.0
        vol = float(np.std(window_ret, ddof=1)) * np.sqrt(252) * 100.0
        rows.append((closes.index[i], vol, worst))
    return pd.DataFrame(rows, columns=["date", "fwd_vol_pct", "fwd_worst_dd_pct"]).set_index("date")


def _group_stats(frame: pd.DataFrame, mask: pd.Series) -> dict:
    sub = frame[mask.reindex(frame.index).fillna(False).astype(bool)]
    if sub.empty:
        return {"days": 0, "mean_fwd_vol_pct": None, "mean_fwd_worst_dd_pct": None}
    return {
        "days": int(len(sub)),
        "mean_fwd_vol_pct": round(float(sub["fwd_vol_pct"].mean()), 2),
        "mean_fwd_worst_dd_pct": round(float(sub["fwd_worst_dd_pct"].mean()), 2),
    }


def _compare(risky: dict, calm: dict) -> dict:
    if risky["days"] < MIN_DAYS or calm["days"] < MIN_DAYS:
        return {"result": "insufficient data",
                "reason": f"needs at least {MIN_DAYS} days in each group (have {risky['days']} and {calm['days']})"}
    passes = (risky["mean_fwd_vol_pct"] > calm["mean_fwd_vol_pct"]
              and risky["mean_fwd_worst_dd_pct"] < calm["mean_fwd_worst_dd_pct"])
    return {"result": "pass" if passes else "fail",
            "reason": "higher forward volatility and deeper worst drawdown than the calm group"
            if passes else "did not show both higher forward volatility and a deeper worst drawdown"}


def validate(labels: pd.Series, spy_close: pd.Series, divergence: pd.Series,
             horizon: int = HORIZON_SESSIONS) -> dict:
    """Runs the agreed test. `labels` and `divergence` are indexed by date."""
    risk = forward_risk(spy_close, horizon)
    labels = labels.copy()
    labels.index = pd.to_datetime(labels.index)
    risk.index = pd.to_datetime(risk.index)
    labelled = risk.join(labels.rename("label"), how="inner")
    flags = divergence.copy()
    flags.index = pd.to_datetime(flags.index)
    flagged = risk.join(flags.rename("flag"), how="inner")

    label_groups = {name: _group_stats(labelled, labelled["label"] == name)
                    for name in ("Risk-On", "Constructive", "Neutral", "Cautious", "Risk-Off")}
    risk_off = _group_stats(labelled, labelled["label"].isin(RISK_OFF_LABELS))
    risk_on = label_groups["Risk-On"]
    divergence_days = _group_stats(flagged, flagged["flag"].astype(bool))
    non_divergence = _group_stats(flagged, ~flagged["flag"].astype(bool))

    return {
        "horizon_sessions": horizon,
        "min_days_per_group": MIN_DAYS,
        "caveat": "Forward windows overlap; days are not independent. No significance is claimed.",
        "label_groups": label_groups,
        "label_test": {"risk_off_or_cautious": risk_off, "risk_on": risk_on,
                       **_compare(risk_off, risk_on)},
        "divergence_test": {"flagged": divergence_days, "unflagged": non_divergence,
                            **_compare(divergence_days, non_divergence)},
    }


def divergence_series(spy_close: pd.Series, breadth_50dma: pd.Series, breadth_threshold: float = 35.0) -> pd.Series:
    """Per-date divergence flag, the same rule as regime_dimensions.divergence_flag."""
    spy = spy_close.dropna().astype(float)
    spy_50 = spy.rolling(50).mean()
    breadth = breadth_50dma.reindex(spy.index).ffill()
    return ((spy > spy_50) & (breadth < breadth_threshold)).where(spy_50.notna() & breadth.notna(), False)
