from datetime import datetime, timedelta
from typing import Optional

from .data_service import get_latest_price
from .fund_comparison_service import price_near_date

DEFAULT_LOOKBACK_DAYS = 30


def compute_portfolio_performance(positions: list[dict], lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    """
    For each position (ticker + shares), compares today's live value against
    what those same shares were worth `lookback_days` ago — an
    apples-to-apples "how has my portfolio actually moved" read, independent
    of whatever current_price happened to be stored at the last save.
    """
    when = datetime.utcnow() - timedelta(days=lookback_days)
    rows = []
    total_now = 0.0
    total_then = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos.get("shares") or 0
        if shares <= 0:
            continue

        price_now = get_latest_price(ticker)
        price_then = price_near_date(ticker, when)
        if price_now is None or price_then is None:
            continue

        value_now = shares * price_now
        value_then = shares * price_then
        diff = value_now - value_then
        diff_pct = (diff / value_then * 100.0) if value_then > 0 else None

        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "price_now": price_now,
                "price_30d_ago": price_then,
                "value_now": value_now,
                "value_30d_ago": value_then,
                "diff": diff,
                "diff_pct": diff_pct,
            }
        )
        total_now += value_now
        total_then += value_then

    total_diff = total_now - total_then
    total_diff_pct: Optional[float] = (total_diff / total_then * 100.0) if total_then > 0 else None

    return {
        "lookback_days": lookback_days,
        "rows": rows,
        "total_value_now": total_now,
        "total_value_30d_ago": total_then,
        "value_diff": total_diff,
        "value_diff_pct": total_diff_pct,
    }
