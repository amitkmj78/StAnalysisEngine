from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

from .data_service import get_effective_price, get_extended_hours_price, get_latest_price
from .fund_comparison_service import price_near_date

DEFAULT_LOOKBACK_DAYS = 30
# Each position needs a few independent, I/O-bound yfinance calls, so fan
# them out across positions rather than serializing behind a slow/rate-limited
# ticker. Was 10 — a real, observed trigger for sustained Yahoo rate
# limiting when fanned out with no pacing between workers (see
# stock_finder_service.py's MAX_PARALLEL_FETCHES for the incident).
MAX_PARALLEL_FETCHES = 4


def compute_total_portfolio_value(positions: list[dict]) -> float:
    """
    Just today's live total — no historical lookback fetch, unlike
    compute_portfolio_performance below. Used wherever only the current
    number matters (e.g. strategy plan progress tracking) so that call
    site isn't paying for a price-30-days-ago lookup it doesn't need.

    Uses get_effective_price, so this reflects an after-hours move
    (earnings, news) once the market is in one of those states, not the
    stale regular-session close until the next open.
    """
    total = 0.0
    for pos in positions:
        shares = pos.get("shares") or 0
        if shares <= 0:
            continue
        price_now = get_effective_price(pos["ticker"])
        if price_now is not None:
            total += shares * price_now
    return total


def _compute_position_row(pos: dict, when: datetime) -> Optional[dict]:
    ticker = pos["ticker"]
    shares = pos.get("shares") or 0
    avg_cost = pos.get("avg_cost")
    if shares <= 0:
        return None

    cost_basis = shares * avg_cost if avg_cost else None

    price_now_regular = get_latest_price(ticker)
    price_then = price_near_date(ticker, when)
    extended_hours = get_extended_hours_price(ticker)
    # Prefer the after-hours quote when the market is actually in one of
    # those states — see get_effective_price. price_now_regular is kept
    # separately so the UI can still show "regular session: $X" next to
    # the after-hours-based value it's now actually using.
    price_now = extended_hours["price"] if extended_hours else price_now_regular

    if price_now is None:
        return {
            "ticker": ticker,
            "shares": shares,
            "avg_cost": avg_cost,
            "cost_basis": cost_basis,
            "price_now": None,
            "price_now_regular": None,
            "price_30d_ago": None,
            "value_now": None,
            "value_30d_ago": None,
            "diff": None,
            "diff_pct": None,
            "gain_vs_cost": None,
            "gain_vs_cost_pct": None,
            "price_unavailable": True,
            "extended_hours": None,
            "used_extended_hours": False,
        }

    value_now = shares * price_now
    value_then = shares * price_then if price_then is not None else None
    diff = value_now - value_then if value_then is not None else None
    diff_pct = (diff / value_then * 100.0) if value_then not in (None, 0) else None

    gain_vs_cost = value_now - cost_basis if cost_basis is not None else None
    gain_vs_cost_pct = (gain_vs_cost / cost_basis * 100.0) if cost_basis not in (None, 0) else None

    return {
        "ticker": ticker,
        "shares": shares,
        "avg_cost": avg_cost,
        "cost_basis": cost_basis,
        "price_now": price_now,
        "price_now_regular": price_now_regular,
        "price_30d_ago": price_then,
        "value_now": value_now,
        "value_30d_ago": value_then,
        "diff": diff,
        "diff_pct": diff_pct,
        "gain_vs_cost": gain_vs_cost,
        "gain_vs_cost_pct": gain_vs_cost_pct,
        "price_unavailable": False,
        "extended_hours": extended_hours,
        "used_extended_hours": extended_hours is not None,
    }


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

    Each position's fetches (current price, price `lookback_days` ago,
    extended-hours price) run in parallel across positions — same pattern
    as stock_finder_service/entry_strategy_service — so one rate-limited
    or slow ticker doesn't serialize the whole portfolio's load behind it.
    """
    when = datetime.utcnow() - timedelta(days=lookback_days)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        rows = [
            row
            for row in executor.map(lambda pos: _compute_position_row(pos, when), positions)
            if row is not None
        ]

    total_now = sum(r["value_now"] for r in rows if r["value_now"] is not None)
    total_then = sum(r["value_30d_ago"] for r in rows if r["value_30d_ago"] is not None)
    total_cost_basis = sum(r["cost_basis"] for r in rows if r["cost_basis"] is not None)

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
