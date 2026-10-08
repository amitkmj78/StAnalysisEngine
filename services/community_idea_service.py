"""COM-1/2/4: pure logic for user-published trade ideas -- scoring at
horizon against SPY, and the leaderboard's risk-adjusted ranking.

COM-2's scoring reuses services/stock_detail_service.py::
evaluate_signal_outcome UNMODIFIED: an idea's `direction` (LONG/SHORT)
maps onto that function's `signal` parameter exactly ("Buy" wins if the
stock rose, "Trim" wins if it didn't -- LONG/SHORT need the identical
rule), so there is no second hit/miss implementation to keep in sync
with DET-3's.

COM-4's ranking is NOT services/backtest_engine.py::sharpe() -- that
function annualizes a DAILY RETURN TIME SERIES (assumes periods_per_year
trading days), the wrong unit for a set of independent, point-in-time
idea bets. risk_adjusted_excess_return below is a small new pure
function with the same *shape* services/challenge_service.py::score_for
already established (None -- not zero -- below a minimum sample,
because a few lucky/unlucky ideas are mostly noise, same reasoning
challenge_service.py's MIN_DAYS_FOR_RISK_SCORE applies to a different
kind of sample).
"""

from __future__ import annotations

import statistics
from typing import Optional

import pandas as pd

from services.stock_detail_service import evaluate_signal_outcome

DIRECTIONS = {"LONG", "SHORT"}
_DIRECTION_TO_SIGNAL = {"LONG": "Buy", "SHORT": "Trim"}

# COM-4: an author needs at least this many SCORED ideas before a
# risk-adjusted rank is shown -- below it, the leaderboard still shows
# their return/sample size (COM-4's own acceptance text), just no score,
# and they sort last rather than at a misleadingly favorable rank.
MIN_IDEAS_FOR_LEADERBOARD = 10


def evaluate_idea_outcome(as_of_date, direction: str, closes: pd.Series, horizon_days: int) -> Optional[dict]:
    """Thin adapter over evaluate_signal_outcome -- translates COM-1's
    LONG/SHORT vocabulary into DET-3's Buy/Trim vocabulary and nothing
    else. Returns the exact same shape (entry_date, exit_date,
    entry_price, exit_price, realized_return_pct, outcome), or None
    when the horizon hasn't elapsed yet (never guessed at)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(DIRECTIONS)}")
    return evaluate_signal_outcome(as_of_date, _DIRECTION_TO_SIGNAL[direction], closes, horizon_days)


def risk_adjusted_excess_return(excess_returns_pct: list[float], min_samples: int = MIN_IDEAS_FOR_LEADERBOARD) -> Optional[float]:
    """Mean excess-return-vs-SPY divided by its own standard deviation,
    across an author's scored ideas -- a Sharpe-STYLE ratio over
    cross-idea bets, not a time series (no annualization: there's no
    "periods per year" for a set of ideas with different horizons).
    None (never 0) below `min_samples` -- not enough data to say
    anything, not "this author scores zero". None (never a divide-by-
    zero crash) when every idea had the identical excess return (zero
    variance) -- that's "undefined", not infinite."""
    n = len(excess_returns_pct)
    if n < min_samples:
        return None
    mean = statistics.mean(excess_returns_pct)
    stdev = statistics.pstdev(excess_returns_pct)
    if stdev == 0:
        return None
    return round(mean / stdev, 4)


def leaderboard_sort_key(score: Optional[float]) -> tuple[bool, float]:
    """Same (score is None, -(score or 0)) convention
    services/challenge_leaderboard.py::build_leaderboard already uses:
    scored entries first (ranked by score desc), unscored entries
    always last, regardless of how the caller's own sort is stable."""
    return (score is None, -(score or 0.0))


def worst_idea(scored_ideas: list[dict]) -> Optional[dict]:
    """COM-3: the single worst idea in an author's record -- the most
    adverse move against the idea's own direction (a LONG that fell
    hardest, or a SHORT that rose hardest), same per-direction
    "worst miss" convention services/signal_publication_service.py::
    worst_misses already generalized for Buy/Trim across the whole
    universe (FND-3)."""
    if not scored_ideas:
        return None

    def _key(idea: dict) -> float:
        r = idea["realized_return_pct"]
        return r if idea["direction"] != "SHORT" else -r

    return min(scored_ideas, key=_key)
