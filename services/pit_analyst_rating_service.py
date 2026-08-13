import logging

import yfinance as yf

from .analyst_rating_service import _parse_analyst_info
from .rate_limit_utils import fetch_with_backoff
from .stock_finder_service import _universe_tickers

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
SOURCE = "yfinance"


def capture_universe_analyst_ratings(universe_id: str = DEFAULT_UNIVERSE) -> list[dict]:
    """
    One row per ticker that currently has analyst coverage, for the same
    real, third-party consensus + price targets shown on the Stock
    Screener's "Analyst Rating" column (see analyst_rating_service.py),
    captured as of right now. Tickers with no coverage (ETFs, funds, small
    caps) are simply absent from the result, not a null-filled row.

    Fetches the raw .info directly (via fetch_with_backoff, for the
    rate-limit retry) rather than calling get_analyst_rating_summary,
    since that function swallows its own exceptions and would give
    fetch_with_backoff nothing to retry on — see _parse_analyst_info for
    the shared field-mapping logic both call sites use. Read-only —
    doesn't write to the DB, callers persist.
    """
    tickers = list(_universe_tickers(universe_id))
    rows = []
    for ticker in tickers:
        try:
            info = fetch_with_backoff(lambda t=ticker: yf.Ticker(t).info)
        except Exception as e:
            logger.warning("PIT analyst rating capture: failed for %s: %s", ticker, e)
            continue
        summary = _parse_analyst_info(ticker, info)
        if summary is None:
            continue
        rows.append({**summary, "source": SOURCE})
    return rows
