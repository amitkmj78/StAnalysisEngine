import logging

import yfinance as yf

from .rate_limit_utils import fetch_with_backoff
from .stock_finder_service import _safe_percent, _universe_tickers

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
SOURCE = "yfinance"


def capture_universe_fundamentals(universe_id: str = DEFAULT_UNIVERSE) -> list[dict]:
    """
    TR-3 Phase 3: one row per ticker for the fundamental inputs to the
    "Long Term" composite score (forward_pe, revenue_growth, earnings_growth
    — see stock_finder_service.GOAL_WEIGHTS), captured as of right now.
    Reuses _safe_percent so the growth-rate transform matches the composite
    score exactly, not a lookalike reimplementation that could drift.
    yfinance has no batch equivalent of .info, so this is one request per
    ticker; a single ticker's failure is skipped rather than aborting the
    whole capture. Read-only — doesn't write to the DB, callers persist.

    Uses _universe_tickers (not the raw STOCK_UNIVERSES dict) — see the
    matching comment in pit_price_service.py for why.
    """
    tickers = list(_universe_tickers(universe_id))
    rows = []
    for ticker in tickers:
        try:
            info = fetch_with_backoff(lambda t=ticker: yf.Ticker(t).info) or {}
        except Exception as e:
            logger.warning("PIT fundamentals capture: failed for %s: %s", ticker, e)
            continue
        rows.append(
            {
                "ticker": ticker,
                "forward_pe": info.get("forwardPE"),
                "revenue_growth_pct": _safe_percent(info.get("revenueGrowth")),
                "earnings_growth_pct": _safe_percent(info.get("earningsGrowth")),
                "source": SOURCE,
            }
        )
    return rows
