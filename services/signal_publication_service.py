import logging
import subprocess
from functools import lru_cache

from .data_service import get_latest_price
from .prediction_service import generate_trading_signal, predict_future_prices
from .stock_finder_service import get_stock_finder_table

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TOP_N = 25
DEFAULT_PREDICT_PERIOD = "1y"
DEFAULT_PREDICT_DAYS_AHEAD = 10
# Selectable horizons for comparing the Predict-page algorithm against the
# published momentum picks. Capped at 30 to match the published lookback
# window (comparing beyond that would forecast further out than the
# momentum return it's being set against).
PREDICT_COMPARE_HORIZONS = [1, 5, 10, 30]


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


def compute_predict_algo_comparison(
    tickers: list[str],
    period: str = DEFAULT_PREDICT_PERIOD,
    days_ahead: int = DEFAULT_PREDICT_DAYS_AHEAD,
) -> list[dict]:
    """
    For each ticker, runs the same trained-model algorithm used on /predict
    (predict_future_prices + generate_trading_signal) — a completely
    different, non-deterministic signal from the momentum ranking above —
    so a reader can see what that separate model currently says about
    today's published picks.

    Only valid against the *current* (latest) publication: this always
    reflects today's price data, so running it against an older published
    date would silently use data the model couldn't have had at that
    original date — there's no point-in-time store yet to prevent that
    honestly. Callers must not offer this for historical dates.
    """
    rows = []
    for ticker in tickers:
        last_close = get_latest_price(ticker)
        future_df = predict_future_prices(ticker, period, days_ahead, False)
        if last_close is None or future_df is None or future_df.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "predict_signal": None,
                    "predict_expected_return_pct": None,
                    "predict_target_price": None,
                }
            )
            continue

        sig = generate_trading_signal(last_close, future_df)
        rows.append(
            {
                "ticker": ticker,
                "predict_signal": sig.get("signal"),
                "predict_expected_return_pct": sig.get("expected_return_pct"),
                "predict_target_price": sig.get("target_price"),
            }
        )
    return rows
