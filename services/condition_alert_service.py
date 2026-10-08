"""ALX-1: live condition-builder alerts -- combine price, indicator,
score, signal, regime and earnings conditions with AND/OR.

Lifts services/strategy_engine.py's field vocabulary and rule_mask
(including its crosses_above/crosses_below semantics) rather than
reimplementing a DSL -- that module already built and tested the exact
comparison logic this needs, just for one ticker's full backtest history.
This module adds the fields a backtest rule never needed (the live price,
the two-score system's score/signal) and evaluates the LATEST row only,
not a full walk-forward.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from services.strategy_engine import (
    CATEGORY_OPS,
    NUMERIC_FIELDS as _BACKTEST_NUMERIC_FIELDS,
    NUMERIC_OPS,
    REGIME_FIELD,
    REGIME_LABELS,
    Rule,
    feature_frame,
    rule_mask,
)

# price/score on top of strategy_engine's indicator/earnings vocabulary --
# fields a live alert condition can reference that a backtest rule never
# needed (a backtest already knows its own entry price; a live alert has
# to ask for "the latest price" explicitly).
PRICE_FIELD = "price"
NUMERIC_FIELDS = {
    **_BACKTEST_NUMERIC_FIELDS,
    PRICE_FIELD: "Latest price ($)",
    "short_score": "Short-Term Score",
    "long_score": "Long-Term Score",
}
SIGNAL_VALUES = ("Buy", "Hold", "Trim")
SIGNAL_FIELDS = {"short_signal": "Short-Term Signal", "long_signal": "Long-Term Signal"}
CATEGORY_FIELDS = {REGIME_FIELD: REGIME_LABELS, **{k: SIGNAL_VALUES for k in SIGNAL_FIELDS}}

COMBINATORS = {"AND", "OR"}
MAX_CONDITIONS = 5

# ALX-2: score/signal/regime/earnings fields are themselves daily
# snapshots (stock_scores/market_regime_daily update once a day,
# earnings reports happen on their own schedule) -- re-checking them
# every minute would just re-read the same unchanged value until the
# next day's capture. Only alerts made ENTIRELY of the remaining
# price/indicator fields are eligible for the faster intraday job;
# everything else stays on the daily-cadence job.
DAILY_ONLY_FIELDS = {"sessions_since_earnings", REGIME_FIELD, "short_score", "long_score", "short_signal", "long_signal"}


def is_intraday_eligible(conditions: list[Rule]) -> bool:
    return all(c.field not in DAILY_ONLY_FIELDS for c in conditions)


def parse_condition(raw: dict) -> Rule:
    """Validates one condition against ALX-1's extended vocabulary.
    strategy_engine.Rule.parse() only knows its own backtest fields, so
    this is a sibling validator, not an override -- it builds the SAME
    Rule dataclass, so rule_mask (the actual comparison logic) below is
    reused unmodified."""
    field_, op, value = raw.get("field"), raw.get("op"), raw.get("value")
    if field_ in CATEGORY_FIELDS:
        if op not in CATEGORY_OPS or value not in CATEGORY_FIELDS[field_]:
            raise ValueError(f"{field_} compares to one of {CATEGORY_FIELDS[field_]} with 'is' or 'is_not'")
    elif field_ in NUMERIC_FIELDS:
        if op not in NUMERIC_OPS:
            raise ValueError(f"{field_} supports >, <, >=, <=, crosses_above, crosses_below")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field_} needs a number")
    else:
        raise ValueError(f"unknown field: {field_}")
    return Rule(field_, op, value)


def parse_conditions(raw_list: list[dict]) -> list[Rule]:
    if not raw_list:
        raise ValueError("add at least one condition")
    if len(raw_list) > MAX_CONDITIONS:
        raise ValueError(f"at most {MAX_CONDITIONS} conditions")
    return [parse_condition(r) for r in raw_list]


def build_condition_frame(
    prices: pd.DataFrame,
    regime_by_date: Optional[dict[str, str]] = None,
    earnings_reports: Optional[list] = None,
    score_history: Optional[list[dict]] = None,
) -> pd.DataFrame:
    """Extends strategy_engine.feature_frame with the fields a live
    condition alert can reference that a backtest rule never needed.

    `score_history`: this ticker's own stock_scores history, as
    [{"as_of_date": date, "short_score", "long_score", "short_signal",
    "long_signal"}] (same shape fetch_latest_scores' underlying rows
    already have). Forward-filled onto every price-history trading day
    -- scores/signals are daily snapshots, not a continuous series, same
    reasoning regime_by_date's own daily grid already relies on."""
    frame = feature_frame(prices, regime_by_date, earnings_reports)
    frame[PRICE_FIELD] = frame["close"]

    by_date = {}
    for row in score_history or []:
        d = row["as_of_date"]
        key = d.isoformat() if hasattr(d, "isoformat") else str(d)
        by_date[key] = row
    keys = [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in frame.index]
    for col in ("short_score", "long_score", "short_signal", "long_signal"):
        frame[col] = [by_date[k].get(col) if k in by_date else None for k in keys]
        frame[col] = frame[col].ffill()
    return frame


def evaluate_condition_alert(conditions: list[Rule], combinator: str, frame: pd.DataFrame) -> Optional[bool]:
    """True/False once every condition's latest value is resolvable;
    None when it can't be said yet (missing column, not enough history
    for an indicator, or a score/regime day that hasn't been captured
    yet) -- an honest "not due to fire," never guessed at. Mirrors this
    app's consistent "return None, don't pad" convention (e.g.
    services.stock_detail_service.evaluate_signal_outcome).

    Checks the SOURCE column's latest value(s) for null before trusting
    rule_mask's boolean output, rather than checking the mask's own
    result -- a plain `series > value` comparison against NaN/None
    returns False, not NaN (standard pandas/numpy comparison semantics),
    so the mask itself never signals "unknown" on its own."""
    if frame.empty or combinator not in COMBINATORS:
        return None
    results = []
    for rule in conditions:
        if rule.field not in frame.columns:
            return None
        series = frame[rule.field]
        if rule.op in ("crosses_above", "crosses_below"):
            if len(series) < 2 or pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-2]):
                return None
        elif pd.isna(series.iloc[-1]):
            return None
        mask = rule_mask(frame, rule)
        results.append(bool(mask.iloc[-1]))
    return all(results) if combinator == "AND" else any(results)
