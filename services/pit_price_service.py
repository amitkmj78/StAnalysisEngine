import logging

import yfinance as yf

from .stock_finder_service import _universe_tickers

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
SOURCE = "yfinance"


def capture_universe_closes(universe_id: str = DEFAULT_UNIVERSE) -> list[dict]:
    """
    TR-3 Phase 1: one row per ticker for the most recently completed trading
    day, as observed right now. Read-only — doesn't write to the DB itself,
    callers persist the result. A caller inserting this with ON CONFLICT DO
    NOTHING on (ticker, price_date) is what makes the store point-in-time:
    once a row exists, it proves this exact close was on record at
    captured_at_utc, immune to any later data-vendor revision.

    Uses _universe_tickers (not the raw STOCK_UNIVERSES dict) because
    "All" and "US - S&P 500" resolve lazily — see stock_finder_service.py.
    Reading STOCK_UNIVERSES directly here silently returned an empty
    placeholder list for those two keys after the S&P 500 expansion,
    which is exactly the bug this comment is here to prevent recurring.
    """
    tickers = list(_universe_tickers(universe_id))
    if not tickers:
        return []

    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:
        logger.warning("PIT capture: yfinance download failed: %s", e)
        return []

    rows = []
    for ticker in tickers:
        try:
            series = raw[ticker]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
        except Exception:
            continue
        if series.empty:
            continue
        last_date = series.index[-1]
        rows.append(
            {
                "ticker": ticker,
                "price_date": last_date.date() if hasattr(last_date, "date") else last_date,
                "close": round(float(series.iloc[-1]), 4),
                "source": SOURCE,
            }
        )
    return rows
