from datetime import datetime, timedelta
from typing import Optional

from .data_service import get_latest_price
from .fund_comparison_service import price_near_date

DEFAULT_LOOKBACK_DAYS = 30


def compute_portfolio_performance(positions: list[dict], lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    """
    For each position (ticker + shares + avg_cost), compares today's live
    value against what those same shares were worth `lookback_days` ago, and
    against what was actually paid for them — both apples-to-apples reads
    against a live-fetched current price, independent of whatever
    current_price happened to be stored at the last save.

    Every saved position gets a row even if a price can't be found for it
    (delisted/mistyped ticker, a fund yfinance doesn't carry, etc.) — the
    row just carries nulls plus `price_unavailable: True` rather than
    silently vanishing from the table.
    """
    when = datetime.utcnow() - timedelta(days=lookback_days)
    rows = []
    total_now = 0.0
    total_then = 0.0
    total_cost_basis = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos.get("shares") or 0
        avg_cost = pos.get("avg_cost")
        if shares <= 0:
            continue

        cost_basis = shares * avg_cost if avg_cost else None
        if cost_basis is not None:
            total_cost_basis += cost_basis

        price_now = get_latest_price(ticker)
        price_then = price_near_date(ticker, when)

        if price_now is None:
            rows.append(
                {
                    "ticker": ticker,
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "cost_basis": cost_basis,
                    "price_now": None,
                    "price_30d_ago": None,
                    "value_now": None,
                    "value_30d_ago": None,
                    "diff": None,
                    "diff_pct": None,
                    "gain_vs_cost": None,
                    "gain_vs_cost_pct": None,
                    "price_unavailable": True,
                }
            )
            continue

        value_now = shares * price_now
        value_then = shares * price_then if price_then is not None else None
        diff = value_now - value_then if value_then is not None else None
        diff_pct = (diff / value_then * 100.0) if value_then not in (None, 0) else None

        gain_vs_cost = value_now - cost_basis if cost_basis is not None else None
        gain_vs_cost_pct = (gain_vs_cost / cost_basis * 100.0) if cost_basis not in (None, 0) else None

        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "avg_cost": avg_cost,
                "cost_basis": cost_basis,
                "price_now": price_now,
                "price_30d_ago": price_then,
                "value_now": value_now,
                "value_30d_ago": value_then,
                "diff": diff,
                "diff_pct": diff_pct,
                "gain_vs_cost": gain_vs_cost,
                "gain_vs_cost_pct": gain_vs_cost_pct,
                "price_unavailable": False,
            }
        )
        total_now += value_now
        if value_then is not None:
            total_then += value_then

    total_diff = total_now - total_then
    total_diff_pct: Optional[float] = (total_diff / total_then * 100.0) if total_then > 0 else None

    total_gain_vs_cost = total_now - total_cost_basis
    total_gain_vs_cost_pct: Optional[float] = (
        (total_gain_vs_cost / total_cost_basis * 100.0) if total_cost_basis > 0 else None
    )

    return {
        "lookback_days": lookback_days,
        "rows": rows,
        "total_value_now": total_now,
        "total_value_30d_ago": total_then,
        "value_diff": total_diff,
        "value_diff_pct": total_diff_pct,
        "total_cost_basis": total_cost_basis,
        "total_gain_vs_cost": total_gain_vs_cost,
        "total_gain_vs_cost_pct": total_gain_vs_cost_pct,
    }
