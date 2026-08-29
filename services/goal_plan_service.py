from concurrent.futures import ThreadPoolExecutor, as_completed

from services.cache_utils import ttl_cache
from services.yfinance_cache import get_cached_history

# Was 10 — a real, observed trigger for sustained Yahoo rate limiting when
# fanned out with no pacing between workers (see stock_finder_service.py's
# MAX_PARALLEL_FETCHES for the incident).
MAX_PARALLEL_FETCHES = 4
# Every held ticker keeps at least this share of new money regardless of
# how weak its current signal is — this tilts where a recurring
# contribution goes, it doesn't abandon a holding entirely over one
# short-term reading.
MIN_ALLOCATION_WEIGHT_FLOOR = 0.5


@ttl_cache(maxsize=256, ttl_seconds=3600)
def get_annualized_return_pct(ticker: str, years: int = 3) -> float | None:
    """
    Trailing CAGR over `years` of daily closes — a long-horizon return
    estimate, deliberately separate from the Portfolio page's short-term
    (10-day) quant forecast. Compounding a multi-year goal projection off
    a 10-day forecast would extrapolate a short-term wiggle into an
    absurd annual rate; this uses actual multi-year trailing performance
    instead, the same approach the Monthly Investing Plan tool's
    "3Y Annualized %" already uses.
    """
    try:
        hist = get_cached_history(ticker, f"{years}y", auto_adjust=True)
    except Exception:
        return None
    if hist.empty:
        return None
    close = hist["Close"]
    if len(close) < 2:
        return None
    actual_years = len(close) / 252
    if actual_years <= 0:
        return None
    start, end = float(close.iloc[0]), float(close.iloc[-1])
    if start <= 0:
        return None
    return ((end / start) ** (1 / actual_years) - 1.0) * 100


def get_annualized_returns(tickers: list[str], years: int = 3) -> dict[str, float | None]:
    unique = list(dict.fromkeys(tickers))
    results: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = {executor.submit(get_annualized_return_pct, t, years): t for t in unique}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def build_signal_weighted_allocation(rows: list[dict]) -> list[dict]:
    """
    rows: [{"ticker": str, "expected_return_pct": float | None, ...}, ...]
    — expected_return_pct is the Portfolio page's existing short-term
    (10-day) quant signal per holding.

    Splits 100% across tickers, tilted toward whichever currently has the
    strongest expected return, while keeping every ticker at least
    MIN_ALLOCATION_WEIGHT_FLOOR percent so a temporarily weak signal
    doesn't fully zero out a holding's share of new money. Tickers with
    no expected_return_pct (forecast unavailable) get exactly the floor
    and are excluded from the signal-strength weighting.

    Returns the same rows with a "weight_pct" field added, summing to 100.
    """
    if not rows:
        return []

    scored = [r for r in rows if r.get("expected_return_pct") is not None]
    unscored = [r for r in rows if r.get("expected_return_pct") is None]

    if not scored:
        pct = 100.0 / len(rows)
        return [{**r, "weight_pct": round(pct, 2)} for r in rows]

    min_return = min(r["expected_return_pct"] for r in scored)
    shifted = {
        id(r): (r["expected_return_pct"] - min_return) + MIN_ALLOCATION_WEIGHT_FLOOR
        for r in scored
    }
    total_shifted = sum(shifted.values())

    reserved_pct = MIN_ALLOCATION_WEIGHT_FLOOR * len(unscored)
    remaining_pct = max(0.0, 100.0 - reserved_pct)

    out = []
    for r in scored:
        weight = (shifted[id(r)] / total_shifted) * remaining_pct if total_shifted else 0.0
        out.append({**r, "weight_pct": weight})
    for r in unscored:
        out.append({**r, "weight_pct": MIN_ALLOCATION_WEIGHT_FLOOR})

    total = sum(o["weight_pct"] for o in out) or 1.0
    for o in out:
        o["weight_pct"] = round(o["weight_pct"] / total * 100.0, 2)
    return out


def solve_goal_plan(
    current_value: float,
    target_amount: float,
    months: int,
    current_holdings_annualized_return_pct: float | None,
    contribution_annualized_return_pct: float | None,
    monthly_amount: float | None = None,
) -> dict:
    """
    Two separate compounding streams, since new money and existing money
    grow at different blended rates (current holdings vs. where a
    recurring contribution is allocated):
      1. current_value compounds at current_holdings_annualized_return_pct
         (blended by today's $ weight across holdings) for `months`.
      2. A monthly contribution, if any, compounds as an annuity-due at
         contribution_annualized_return_pct (blended by allocation weight).

    required_monthly_contribution solves stream 2 algebraically for the
    amount needed so stream 1 + stream 2 == target_amount. 0 if the
    current holdings are already projected to clear the target on their
    own; None if the target date is not in the future.
    """
    r_current = (current_holdings_annualized_return_pct or 0.0) / 100.0 / 12
    r_contrib = (contribution_annualized_return_pct or 0.0) / 100.0 / 12

    future_value_of_current = current_value * ((1 + r_current) ** months) if months > 0 else current_value

    def fv_of_contributions(amount: float) -> float:
        if amount <= 0 or months <= 0:
            return 0.0
        if r_contrib == 0:
            return amount * months
        return amount * (((1 + r_contrib) ** months - 1) / r_contrib) * (1 + r_contrib)

    shortfall = target_amount - future_value_of_current
    if shortfall <= 0:
        required_monthly: float | None = 0.0
    elif months <= 0:
        required_monthly = None
    elif r_contrib == 0:
        required_monthly = shortfall / months
    else:
        annuity_factor = (((1 + r_contrib) ** months - 1) / r_contrib) * (1 + r_contrib)
        required_monthly = shortfall / annuity_factor if annuity_factor else None

    result = {
        "future_value_of_current_holdings": future_value_of_current,
        "required_monthly_contribution": required_monthly,
    }
    if monthly_amount is not None:
        projected_total = future_value_of_current + fv_of_contributions(monthly_amount)
        result["projected_value_with_given_contribution"] = projected_total
        result["gap_vs_target"] = projected_total - target_amount
    return result
