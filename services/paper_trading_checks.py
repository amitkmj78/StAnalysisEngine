"""
Pure pre-trade checks for a paper-trading order ticket (TRD-15 through
TRD-19). No I/O -- every function takes plain numbers/dicts already
fetched by the caller (the router), so these are trivially unit-testable
with synthetic data and can never accidentally call Alpaca or the DB
themselves.

The router runs every check and collects every failure (not just the
first), so the ticket's review screen can show every problem at once
rather than one at a time across repeated submit attempts.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    code: str
    message: str


def estimate_order_value(qty: float, order_type: str, limit_price: Optional[float], last_price: Optional[float]) -> Optional[float]:
    """Estimated notional value of the order: limit price when the order
    carries a fixed limit (limit or stop_limit -- the price the user
    actually set), else the last trade price (market, stop, trailing_stop
    all have no fixed price before they fire). None if neither price is
    available -- callers should treat that as "can't estimate," not as a
    zero-value order."""
    price = limit_price if order_type in ("limit", "stop_limit") and limit_price is not None else last_price
    if price is None:
        return None
    return qty * price


def compute_risk_based_quantity(entry_price: float, stop_price: float, max_loss_dollars: float) -> Optional[float]:
    """BRK-3: shares = max loss ($) / |entry - stop|, rounded down to a
    whole share (same floor-to-whole-share convention paper order qty
    uses elsewhere). None when entry and stop are equal (division by
    zero -- there's no stop distance to size against) or either input is
    non-positive."""
    if entry_price <= 0 or stop_price <= 0 or max_loss_dollars <= 0:
        return None
    distance = abs(entry_price - stop_price)
    if distance == 0:
        return None
    return float(math.floor(max_loss_dollars / distance))


def check_bracket_order_sanity(
    side: str, entry_estimate: Optional[float], take_profit_price: Optional[float], stop_loss_price: Optional[float]
) -> CheckResult:
    """BRK-2: a bracket order's take-profit and stop-loss legs must sit on
    the correct sides of the entry price -- Alpaca will reject a
    nonsensical bracket anyway, but a plain-language check here catches it
    before the round-trip. Skipped (passes) when the entry estimate isn't
    known yet (e.g. no last price available) or either leg is unset --
    this only validates legs that are actually present."""
    if entry_estimate is None or take_profit_price is None or stop_loss_price is None:
        return CheckResult(True, "bracket_sanity", "OK")
    if side == "buy":
        ok = stop_loss_price < entry_estimate < take_profit_price
    else:
        ok = take_profit_price < entry_estimate < stop_loss_price
    if not ok:
        return CheckResult(
            False, "bracket_sanity",
            f"For a {side} bracket order, take-profit (${take_profit_price:,.2f}) and stop-loss "
            f"(${stop_loss_price:,.2f}) must sit on the correct sides of the estimated entry price "
            f"(${entry_estimate:,.2f}).",
        )
    return CheckResult(True, "bracket_sanity", "OK")


def check_buying_power(side: str, qty: float, estimate: Optional[float], account: dict, shares_held: float) -> CheckResult:
    if side == "sell":
        if qty > shares_held:
            return CheckResult(False, "insufficient_shares", f"You hold {shares_held:g} shares but tried to sell {qty:g}.")
        return CheckResult(True, "buying_power", "OK")
    buying_power = float(account.get("buying_power", 0))
    if estimate is not None and estimate > buying_power:
        return CheckResult(
            False, "insufficient_buying_power",
            f"Estimated cost ${estimate:,.2f} exceeds your buying power of ${buying_power:,.2f}.",
        )
    return CheckResult(True, "buying_power", "OK")


def check_order_value_limit(estimate: Optional[float], max_order_value: float) -> CheckResult:
    if estimate is not None and estimate > max_order_value:
        return CheckResult(
            False, "order_value_limit",
            f"Order value ${estimate:,.2f} exceeds your ${max_order_value:,.2f} per-order limit.",
        )
    return CheckResult(True, "order_value_limit", "OK")


def check_portfolio_pct_limit(estimate: Optional[float], portfolio_equity: float, max_pct: float) -> CheckResult:
    if estimate is not None and portfolio_equity > 0 and (estimate / portfolio_equity) * 100 > max_pct:
        pct = (estimate / portfolio_equity) * 100
        return CheckResult(
            False, "portfolio_pct_limit",
            f"This order is {pct:.1f}% of your portfolio, over the {max_pct:g}% per-order limit.",
        )
    return CheckResult(True, "portfolio_pct_limit", "OK")


def check_daily_order_count(orders_today_count: int, max_orders_per_day: int) -> CheckResult:
    if orders_today_count >= max_orders_per_day:
        return CheckResult(
            False, "daily_order_count",
            f"You've reached today's limit of {max_orders_per_day} orders.",
        )
    return CheckResult(True, "daily_order_count", "OK")


def check_price_collar(order_type: str, limit_price: Optional[float], last_price: Optional[float], collar_pct: float) -> CheckResult:
    if order_type != "limit" or limit_price is None or last_price is None or last_price == 0:
        return CheckResult(True, "price_collar", "OK")
    deviation_pct = abs(limit_price - last_price) / last_price * 100
    if deviation_pct > collar_pct:
        return CheckResult(
            False, "price_collar",
            f"Limit price ${limit_price:,.2f} is {deviation_pct:.1f}% away from the last trade price "
            f"${last_price:,.2f}, over the {collar_pct:g}% warning threshold.",
        )
    return CheckResult(True, "price_collar", "OK")


def check_restricted_symbol(ticker: str, restricted_symbols: set) -> CheckResult:
    if ticker.upper() in restricted_symbols:
        return CheckResult(False, "restricted_symbol", f"{ticker.upper()} is on the restricted list and can't be traded.")
    return CheckResult(True, "restricted_symbol", "OK")
