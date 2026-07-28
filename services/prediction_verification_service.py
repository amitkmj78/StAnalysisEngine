from datetime import datetime, timedelta, timezone
from typing import Optional

from .data_service import get_stock_data


def _first_bar_after(ticker: str, after: datetime) -> Optional[tuple]:
    """First (date, open, close) strictly after `after`, or None if not
    enough history has passed yet. Pulls a short recent window — cheap and
    already ttl_cache'd via get_stock_data."""
    data = get_stock_data(ticker, "3mo")
    if data.empty:
        return None
    if after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    for ts, row in data.iterrows():
        ts_utc = ts.to_pydatetime()
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
        if ts_utc > after:
            return ts_utc, float(row["Open"]), float(row["Close"])
    return None


def _signal_correct(signal: Optional[str], last_close: Optional[float], actual_target_price: Optional[float]) -> Optional[bool]:
    if not signal or last_close is None or actual_target_price is None:
        return None
    actual_return = (actual_target_price - last_close) / last_close
    if signal == "BUY":
        return actual_return > 0
    if signal == "SELL":
        return actual_return < 0
    if signal == "HOLD":
        return abs(actual_return) < 0.05
    return None


def verify_prediction(row: dict) -> dict:
    """
    Given a saved_predictions row (dict), returns only the fields that
    should be updated — empty dict if nothing new is verifiable yet.
    Never touches the row's own predicted_at/target_price/etc.
    """
    updates: dict = {}
    ticker = row["ticker"]
    predicted_at = row["predicted_at"]

    if row.get("actual_next_price") is None:
        found = _first_bar_after(ticker, predicted_at)
        if found is not None:
            _, _, actual_next_price = found
            updates["actual_next_price"] = actual_next_price
            if row.get("next_price") is not None and actual_next_price:
                updates["next_price_error_pct"] = round(
                    (row["next_price"] - actual_next_price) / actual_next_price * 100.0, 2
                )

    target_date = row.get("target_date")
    if row.get("actual_target_price") is None and target_date is not None:
        now = datetime.now(timezone.utc)
        target_date_utc = target_date if target_date.tzinfo else target_date.replace(tzinfo=timezone.utc)
        if now >= target_date_utc:
            found = _first_bar_after(ticker, target_date_utc - timedelta(days=1))
            if found is not None:
                _, actual_target_open, actual_target_price = found
                updates["actual_target_open"] = actual_target_open
                updates["actual_target_price"] = actual_target_price
                if row.get("target_price") is not None and actual_target_price:
                    updates["target_price_error_pct"] = round(
                        (row["target_price"] - actual_target_price) / actual_target_price * 100.0, 2
                    )
                updates["signal_correct"] = _signal_correct(
                    row.get("signal"), row.get("last_close"), actual_target_price
                )
                updates["verified_at"] = now

    return updates
