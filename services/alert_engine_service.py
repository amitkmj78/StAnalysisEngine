from typing import Optional

from .data_service import get_latest_price

# Kept as its own module (not stuffed into alert_service.py's old one-off
# alert_breakout helper) so a future condition type has one clear place
# to add a branch -- ALR-1 added score_above/score_below alongside the
# original price_above/price_below.
CONDITION_TYPES = {"price_above", "price_below", "score_above", "score_below"}


def evaluate_alert(
    ticker: str,
    condition_type: str,
    threshold: float,
    latest_short_score: Optional[float] = None,
) -> Optional[float]:
    """Returns the value that satisfied the condition (price or score) if
    it's met right now, else None. price_* conditions fetch a live price
    here, same as always. score_* conditions deliberately do NOT fetch
    anything here -- stock_scores only changes once/day, so the caller
    batch-fetches every ticker's latest short_score in one query and
    passes it in via `latest_short_score` (see web/backend/scheduler.py's
    _evaluate_watchlist_alerts), keeping this function plain, synchronous
    and DB-free, same as the original price-only design."""
    if condition_type in ("price_above", "price_below"):
        price = get_latest_price(ticker)
        if price is None:
            return None
        if condition_type == "price_above" and price >= threshold:
            return price
        if condition_type == "price_below" and price <= threshold:
            return price
        return None
    if condition_type in ("score_above", "score_below"):
        if latest_short_score is None:
            return None
        if condition_type == "score_above" and latest_short_score >= threshold:
            return latest_short_score
        if condition_type == "score_below" and latest_short_score <= threshold:
            return latest_short_score
        return None
    return None
