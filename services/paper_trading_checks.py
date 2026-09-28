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

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    code: str
    message: str


def estimate_order_value(qty: float, order_type: str, limit_price: Optional[float], last_price: Optional[float]) -> Optional[float]:
    """Estimated notional value of the order: limit price when the order
    is a limit order (the price the user actually set), else the last
    trade price for a market order. None if a market order has no last
    price available -- callers should treat that as "can't estimate,"
    not as a zero-value order."""
    price = limit_price if order_type == "limit" and limit_price is not None else last_price
    if price is None:
        return None
    return qty * price


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
