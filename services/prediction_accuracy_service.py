"""
Pure, zero-dependency ranking of saved predictions by ticker accuracy.

No I/O — takes already-fetched (and already auto-verified, per
services/prediction_verification_service.py) saved_predictions rows and
groups/ranks them. Kept dependency-free like services/ranking_utils.py so
it stays importable and unit-testable without a DB connection.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

MIN_VERIFIED_FOR_RECOMMENDATION = 3


def _mean_abs(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(abs(v) for v in values) / len(values), 3)


def compute_prediction_accuracy(
    predictions: list[dict],
    min_verified_for_recommendation: int = MIN_VERIFIED_FOR_RECOMMENDATION,
) -> dict:
    """
    Groups saved predictions by ticker and computes each ticker's win rate
    and average error among predictions with a verdict — a prediction
    counts as "verified" once signal_correct is no longer null (which
    verify_prediction only sets once the target date has actually passed,
    per services/prediction_verification_service.py — a prediction whose
    target date hasn't arrived yet doesn't get a premature verdict).

    Tickers are ranked 1-indexed by win rate (ties broken by verified
    count, more evidence ranking higher) among tickers with at least one
    verified prediction; tickers with zero verified predictions get
    rank=None rather than a misleading last-place slot.

    A ticker is eligible for the "suggested for portfolio" callout only
    once it has min_verified_for_recommendation or more verified
    predictions — avoids suggesting a ticker off a single lucky call. The
    suggestion is the top-ranked eligible ticker, or None if none qualify
    yet.
    """
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for p in predictions:
        by_ticker[p["ticker"]].append(p)

    rows = []
    for ticker, preds in by_ticker.items():
        verified = [p for p in preds if p.get("signal_correct") is not None]
        verified_count = len(verified)
        win_rate = (
            round(sum(1 for p in verified if p["signal_correct"]) / verified_count, 4)
            if verified_count > 0
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "total_predictions": len(preds),
                "verified_count": verified_count,
                "win_rate": win_rate,
                "avg_next_price_error_pct": _mean_abs(
                    [p["next_price_error_pct"] for p in preds if p.get("next_price_error_pct") is not None]
                ),
                "avg_target_price_error_pct": _mean_abs(
                    [p["target_price_error_pct"] for p in preds if p.get("target_price_error_pct") is not None]
                ),
                "eligible_for_recommendation": verified_count >= min_verified_for_recommendation,
            }
        )

    ranked = sorted(
        (r for r in rows if r["verified_count"] > 0),
        key=lambda r: (r["win_rate"], r["verified_count"]),
        reverse=True,
    )
    rank_by_ticker = {r["ticker"]: i + 1 for i, r in enumerate(ranked)}
    for r in rows:
        r["rank"] = rank_by_ticker.get(r["ticker"])

    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else 0))

    eligible = [r for r in ranked if r["eligible_for_recommendation"]]
    if eligible:
        best = eligible[0]
        suggested_ticker = best["ticker"]
        suggested_reason = (
            f"Best track record: {best['win_rate'] * 100:.0f}% of {best['verified_count']} "
            f"verified predictions were correct."
        )
    else:
        suggested_ticker = None
        suggested_reason = None

    return {
        "tickers": rows,
        "suggested_ticker": suggested_ticker,
        "suggested_reason": suggested_reason,
        "min_verified_for_recommendation": min_verified_for_recommendation,
    }
