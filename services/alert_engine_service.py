from typing import Optional

from .data_service import get_latest_price

# Price-only for now, kept as its own module (not stuffed into
# alert_service.py's old one-off alert_breakout helper) so a future
# condition type (RSI, etc.) has one clear place to add a branch.
CONDITION_TYPES = {"price_above", "price_below"}


def evaluate_alert(ticker: str, condition_type: str, threshold: float) -> Optional[float]:
    """Returns the current price if the condition is met right now, else None."""
    price = get_latest_price(ticker)
    if price is None:
        return None
    if condition_type == "price_above" and price >= threshold:
        return price
    if condition_type == "price_below" and price <= threshold:
        return price
    return None
