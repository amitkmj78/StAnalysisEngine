import logging
import time

from .data_service import get_latest_price
from .prediction_service import generate_trading_signal, predict_future_prices
from .stock_finder_service import _universe_tickers

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
PREDICT_PERIOD = "1y"
PREDICT_DAYS_AHEAD = 10
SOURCE = "internal-model"
PACING_DELAY_SECONDS = 0.3


def capture_universe_quant_signals(universe_id: str = DEFAULT_UNIVERSE) -> list[dict]:
    """
    One row per ticker for the same BUY/HOLD/SELL quant signal shown on
    /predict and the Stock Screener's "Quant Signal" column, captured as
    of right now. Unlike every other PIT capture in this app, this trains
    a fresh model per ticker (~1s each, CPU-bound, not just a network
    call) — meant for an off-hours scheduled run, not a request-time path.

    A small pacing delay between tickers (not retry-on-exception, like
    rate_limit_utils.fetch_with_backoff uses elsewhere) — the underlying
    get_stock_data/predict_future_prices calls already swallow their own
    fetch failures internally and return an empty result rather than
    raising, so there's no exception here to catch and retry on; this
    just reduces the odds of tripping the rate limit in the first place
    across ~500 sequential tickers.

    A single ticker's failure (no data, no forecast) is skipped rather
    than aborting the whole capture. Read-only — doesn't write to the DB,
    callers persist.
    """
    tickers = list(_universe_tickers(universe_id))
    rows = []
    for ticker in tickers:
        try:
            last_close = get_latest_price(ticker)
            if last_close is None:
                continue
            future_df = predict_future_prices(ticker, PREDICT_PERIOD, PREDICT_DAYS_AHEAD, False)
            if future_df is None or future_df.empty:
                continue
            sig = generate_trading_signal(last_close, future_df)
            if "target_price" not in sig:
                continue
        except Exception as e:
            logger.warning("PIT quant signal capture: failed for %s: %s", ticker, e)
            continue
        finally:
            time.sleep(PACING_DELAY_SECONDS)

        rows.append(
            {
                "ticker": ticker,
                "signal": sig["signal"],
                "expected_return_pct": sig["expected_return_pct"],
                "target_price": sig["target_price"],
                "last_close": sig["last_close"],
                "source": SOURCE,
            }
        )
    return rows
