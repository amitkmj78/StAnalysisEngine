import logging
import subprocess
from functools import lru_cache

from .stock_finder_service import get_stock_finder_table

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TOP_N = 5


@lru_cache(maxsize=1)
def get_model_version_hash() -> str:
    """
    The publication's model_version_hash (TR-2): the git commit the running
    code was built from. The published Signal Set is a pure, deterministic
    function of PIT price data and this codebase — no separately-trained
    model artifact to version — so the commit hash IS the model version.
    Cached for the process lifetime since it can't change without a redeploy.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning("Could not resolve git commit hash: %s", e)
        return "unknown"


def build_daily_signal_set(
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Deterministic top-N momentum ranking — the same trailing-return sort
    already shown on /top-performers, reused here rather than reimplemented
    so the published record and the live leaderboard are provably the same
    rule. Pure function of PIT price data: no fundamentals, no training, no
    randomness — reconstructible at any date given the same price history.
    """
    col = f"Return {lookback_days}D %"
    df = get_stock_finder_table(universe_id)
    if df.empty or col not in df.columns:
        return []

    ranked = df.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n)
    return [
        {
            "rank": i + 1,
            "ticker": str(row["Ticker"]),
            "trailing_return_pct": round(float(row[col]), 4),
        }
        for i, (_, row) in enumerate(ranked.iterrows())
    ]
