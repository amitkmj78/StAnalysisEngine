"""
BEG-3: the paper-trading-first precondition a self-identified beginner
(users.experience_level == 'beginner') must clear before connecting a real
brokerage account via Plaid. Pure logic over already-fetched values, kept
separate from the routers so both plaid_integration.py (enforcement) and
paper_trading.py (the progress-readout endpoint) share one definition of
"ready" rather than two that could drift apart -- same separation
services/trade_impact_service.py uses for its own pure measures.
"""

from datetime import datetime, timezone
from typing import Optional

MIN_DAYS_OPEN = 30
MIN_TRADES = 10
# The "risk and drawdown" lesson's quiz (web/frontend/lib/lessons.ts) doubles
# as BEG-3's required risk quiz, rather than building a second one.
RISK_QUIZ_LESSON_ID = "risk-and-drawdown"


def compute_readiness(
    earliest_account_created_at: Optional[datetime],
    trades_done: int,
    risk_quiz_done: bool,
) -> dict:
    """`earliest_account_created_at` is the oldest `alpaca_paper_accounts.
    created_at` across all of this user's linked paper accounts (None if
    they have none yet). `trades_done` is a count of real (filled) paper
    orders, not order attempts."""
    if earliest_account_created_at is None:
        days_open = 0
    else:
        created = earliest_account_created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        days_open = max(0, (datetime.now(timezone.utc) - created).days)

    trades_done = max(0, trades_done)
    ready = days_open >= MIN_DAYS_OPEN and trades_done >= MIN_TRADES and risk_quiz_done

    missing = []
    if days_open < MIN_DAYS_OPEN:
        missing.append(f"{MIN_DAYS_OPEN - days_open} more day(s) of paper trading")
    if trades_done < MIN_TRADES:
        missing.append(f"{MIN_TRADES - trades_done} more paper trade(s)")
    if not risk_quiz_done:
        missing.append("the risk quiz")

    return {
        "ready": ready,
        "days_open": days_open,
        "days_required": MIN_DAYS_OPEN,
        "trades_done": trades_done,
        "trades_required": MIN_TRADES,
        "risk_quiz_done": risk_quiz_done,
        "missing": missing,
    }
