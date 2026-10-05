"""DIF-2: a stock's own track record for its short-term signals, beside its chart.

Uses the same outcome rules as the stock page history (services.stock_detail_service): a Buy is a hit
if the stock rose over the horizon, a Trim is a hit if it did not, and a Hold has no verdict. The
horizon is the short-term one (10 sessions). Each hit or miss is also compared with SPY over the same
dates, so the excess return is the stock's move minus the market's.

Groups under MIN_SIGNALS show no figures, only the count, and say "not enough data yet".
"""

from datetime import date
from typing import Optional

import pandas as pd

from services.stock_detail_service import DET3_SHORT_HORIZON_DAYS, evaluate_signal_outcome

MIN_SIGNALS = 30


def _close_on(closes: pd.Series, when: str) -> Optional[float]:
    index = closes.index
    if getattr(index, "tz", None) is not None:
        closes = closes.copy()
        closes.index = index.tz_localize(None)
    target = pd.Timestamp(when)
    exact = closes[closes.index == target]
    if not exact.empty:
        return float(exact.iloc[0])
    after = closes[closes.index >= target]
    return float(after.iloc[0]) if not after.empty else None


def _spy_return(spy: pd.Series, entry_date: str, exit_date: str) -> Optional[float]:
    start, end = _close_on(spy, entry_date), _close_on(spy, exit_date)
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def track_record(signal_rows: list[dict], closes: pd.Series, spy_closes: pd.Series) -> dict:
    """signal_rows: [{"as_of_date": "YYYY-MM-DD", "short_signal": "Buy"|"Hold"|"Trim"}] from the stored history."""
    scored = []
    for row in signal_rows:
        signal = row.get("short_signal")
        if signal not in ("Buy", "Trim"):
            continue
        outcome = evaluate_signal_outcome(date.fromisoformat(row["as_of_date"]), signal, closes, DET3_SHORT_HORIZON_DAYS)
        if outcome is None or outcome["outcome"] is None:
            continue
        spy_ret = _spy_return(spy_closes, outcome["entry_date"], outcome["exit_date"])
        if spy_ret is None:
            continue
        scored.append({
            "as_of_date": row["as_of_date"],
            "signal": signal,
            "outcome": outcome["outcome"],
            "realized_return_pct": outcome["realized_return_pct"],
            "excess_vs_spy_pct": round(outcome["realized_return_pct"] - spy_ret, 2),
        })

    n = len(scored)
    base = {"signals_evaluated": n, "min_signals": MIN_SIGNALS, "horizon_sessions": DET3_SHORT_HORIZON_DAYS,
            "enough_data": n >= MIN_SIGNALS}
    if n < MIN_SIGNALS:
        return {**base, "hit_rate_pct": None, "avg_excess_vs_spy_pct": None, "worst_miss": None,
                "message": f"Not enough data yet: {n} of {MIN_SIGNALS} signals have finished their {DET3_SHORT_HORIZON_DAYS}-session test."}

    hits = sum(1 for s in scored if s["outcome"] == "hit")
    misses = [s for s in scored if s["outcome"] == "miss"]
    # The worst miss is the one that moved most against the signal: a Buy that fell, or a Trim whose stock rose.
    worst = None
    if misses:
        worst = min(misses, key=lambda s: s["realized_return_pct"] if s["signal"] == "Buy" else -s["realized_return_pct"])
    return {
        **base,
        "hit_rate_pct": round(hits / n * 100, 1),
        "avg_excess_vs_spy_pct": round(sum(s["excess_vs_spy_pct"] for s in scored) / n, 2),
        "worst_miss": worst,
        "message": None,
    }
