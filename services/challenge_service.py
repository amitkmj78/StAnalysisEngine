"""PPR-2: monthly paper-trading challenges with a friend leaderboard.

A challenge is a private group (joined via a shareable code, not an
existing friend graph -- this app has none) that ranks members' linked
Alpaca paper-trading accounts (PPR-1) over a date window. The one new
piece of plumbing this needs is a daily equity snapshot per account --
see capture_equity_for_account -- since a leaderboard showing *risk*
needs a return series, not the single live balance paper_trading.py's
own /account endpoint already provides.
"""

import logging
import secrets
from datetime import date
from typing import Optional

import numpy as np
from starlette.concurrency import run_in_threadpool

from services import alpaca_trading_client
from services.alpaca_trading_client import AlpacaTradingError
from services.backtest_engine import DAYS_PER_YEAR, cumulative_pct, max_drawdown_pct, sharpe, sortino
from web.backend.crypto_utils import decrypt_token
from web.backend.db import service_conn

logger = logging.getLogger(__name__)

# Excludes 0/O/1/I/L -- a code meant to be read aloud or typed by hand
# shouldn't hinge on telling those apart.
JOIN_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
JOIN_CODE_LENGTH = 6

MIN_SNAPSHOTS_FOR_PERFORMANCE = 2
# Sharpe, Sortino and Calmar on fewer days than this are mostly noise, so the
# leaderboard shows the return but leaves the risk-adjusted score blank.
MIN_DAYS_FOR_RISK_SCORE = 5

SCORING_METHODS = {
    "return": "Raw return",
    "sharpe": "Sharpe ratio",
    "sortino": "Sortino ratio",
    "calmar": "Calmar (return / max drawdown)",
    "excess_spy": "Excess return vs S&P 500",
}
DEFAULT_SCORING = "return"


def generate_join_code() -> str:
    return "".join(secrets.choice(JOIN_CODE_ALPHABET) for _ in range(JOIN_CODE_LENGTH))


def compute_member_performance(equity_snapshots: list[dict], start_date: date, end_date: date) -> dict:
    """Pure, DB-free: given one account's {"as_of_date", "equity"} rows
    (any order, possibly outside the window), returns this member's
    return/risk over [start_date, end_date]. Returns are computed from
    the first snapshot on/after start_date to the last on/before
    end_date -- NOT from account inception -- so members who linked
    their paper account at different times are still compared on an
    even footing, and an in-progress challenge gets a live "as of
    today" reading rather than nothing until it ends.

    Fewer than MIN_SNAPSHOTS_FOR_PERFORMANCE points in range means no
    day-over-day change can be computed at all -- returns a real
    days_of_data alongside None percentages (disclosed, not faked as
    0%), rather than a misleadingly confident number from almost no
    data.
    """
    in_range = sorted(
        (s for s in equity_snapshots if start_date <= s["as_of_date"] <= end_date),
        key=lambda s: s["as_of_date"],
    )
    days_of_data = len(in_range)

    if days_of_data < MIN_SNAPSHOTS_FOR_PERFORMANCE:
        return {
            "return_pct": None,
            "max_drawdown_pct": None,
            "annualized_volatility_pct": None,
            "days_of_data": days_of_data,
        }

    daily_returns_pct = [
        (in_range[i]["equity"] / in_range[i - 1]["equity"] - 1.0) * 100.0 for i in range(1, days_of_data)
    ]

    volatility_pct: Optional[float] = None
    if len(daily_returns_pct) >= 2:
        volatility_pct = round(float(np.std(daily_returns_pct, ddof=1)) * np.sqrt(DAYS_PER_YEAR), 2)

    total_return = cumulative_pct(daily_returns_pct)
    drawdown = max_drawdown_pct(daily_returns_pct)
    risk_scored = days_of_data >= MIN_DAYS_FOR_RISK_SCORE
    calmar = None
    if risk_scored and total_return is not None and drawdown:
        calmar = round(total_return / abs(drawdown), 2)

    return {
        "return_pct": total_return,
        "max_drawdown_pct": drawdown,
        "annualized_volatility_pct": volatility_pct,
        "sharpe": sharpe(daily_returns_pct, 0.0, DAYS_PER_YEAR) if risk_scored else None,
        "sortino": sortino(daily_returns_pct, 0.0, DAYS_PER_YEAR) if risk_scored else None,
        "calmar": calmar,
        "days_of_data": days_of_data,
    }


def rebase_to_100(points: list[tuple[date, float]]) -> list[dict]:
    """Rebases a (date, value) series so its first value is 100. Input need not
    be sorted; a zero or missing first value gives no series rather than a
    division by zero."""
    ordered = sorted((d, float(v)) for d, v in points if v is not None)
    if not ordered or ordered[0][1] == 0:
        return []
    base = ordered[0][1]
    return [{"date": d.isoformat(), "value": round(v / base * 100.0, 2)} for d, v in ordered]


def score_for(method: str, performance: dict, spy_return_pct: Optional[float]) -> Optional[float]:
    """The single number a leaderboard sorts by. None means "not scored yet",
    which sorts last rather than being guessed."""
    if method == "excess_spy":
        if performance.get("return_pct") is None or spy_return_pct is None:
            return None
        return round(performance["return_pct"] - spy_return_pct, 2)
    if method in ("sharpe", "sortino", "calmar"):
        return performance.get(method)
    return performance.get("return_pct")


async def capture_equity_for_account(account: dict) -> bool:
    """Scheduler entry point for one row of alpaca_paper_accounts.
    Mirrors paper_order_sync.py::sync_positions_and_cash_for_account's
    exact decrypt -> API call -> per-step error isolation shape (same
    account/error states), swapping list_positions for get_account's
    "equity" field. Returns True on a fresh insert, False on a no-op
    (bad key, API error, or already captured today)."""
    account_id = account["id"]

    try:
        secret_key = decrypt_token(account["api_secret_key_encrypted"])
    except ValueError:
        await _mark_account_status(account_id, "invalid_key", "Could not decrypt stored key.")
        return False

    try:
        live = await run_in_threadpool(alpaca_trading_client.get_account, account["api_key_id"], secret_key)
    except AlpacaTradingError as e:
        logger.warning("Alpaca get_account failed for account %s: %s", account_id, e.message)
        if e.status_code in (401, 403):
            await _mark_account_status(account_id, "invalid_key", e.message)
        return False

    equity = live.get("equity")
    if equity is None:
        return False

    async with service_conn() as conn:
        result = await conn.execute(
            """
            INSERT INTO paper_account_equity_snapshots (alpaca_paper_account_id, user_id, as_of_date, equity)
            VALUES ($1, $2::uuid, $3, $4)
            ON CONFLICT (alpaca_paper_account_id, as_of_date) DO NOTHING
            """,
            account_id, account["user_id"], date.today(), float(equity),
        )
    return result == "INSERT 0 1"


async def _mark_account_status(account_id: int, status: str, error_detail: Optional[str]) -> None:
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE alpaca_paper_accounts SET status = $1, last_sync_error = $2 WHERE id = $3",
            status, error_detail, account_id,
        )
