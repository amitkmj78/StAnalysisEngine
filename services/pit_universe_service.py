from .index_fund_service import INDEX_FUND_UNIVERSE
from .screener_service import INDEX_MAP
from .stock_finder_service import SP500_UNIVERSE_NAME, _universe_tickers

# TR-6 reconciliation resolves a target date's ticker list from this
# module's stored snapshots, so it can only ever succeed for a universe_key
# actually captured here. "All" is signal_publication_service.DEFAULT_UNIVERSE
# (what actually gets published/reconciled by default) and SP500_UNIVERSE_NAME
# is the other real, non-sample universe -- both resolved lazily via
# _universe_tickers, same fix already applied once to Phase 1 (see
# pit_price_service.capture_universe_closes's docstring) but never carried
# over to this module: reading INDEX_MAP's static dict directly silently
# missed both keys entirely, since neither is a real INDEX_MAP entry.
_REAL_STOCK_UNIVERSES = ["All", SP500_UNIVERSE_NAME]


def capture_universe_membership() -> list[dict]:
    """
    TR-3 Phase 2: a snapshot of exactly which tickers belong to which
    universe right now — the real "All"/S&P 500 stock universes (resolved
    live via _universe_tickers, same source Phase 1 price capture and
    signal publication use), the small INDEX_MAP sample lists (used
    elsewhere, e.g. the legacy screener), and fund universes
    (INDEX_FUND_UNIVERSE). The "All"/S&P 500 lookup is a live (cached 24h)
    network call, so this is no longer I/O-free — callers already run this
    off the main request path (see web/backend/pit_prices.py's
    run_in_threadpool wrapper).
    """
    rows = []
    for universe_key in _REAL_STOCK_UNIVERSES:
        for ticker in _universe_tickers(universe_key):
            rows.append({"asset_type": "stock", "universe_key": universe_key, "ticker": ticker})
    for universe_key, tickers in INDEX_MAP.items():
        for ticker in tickers:
            rows.append({"asset_type": "stock", "universe_key": universe_key, "ticker": ticker})
    for fund in INDEX_FUND_UNIVERSE:
        rows.append({"asset_type": "fund", "universe_key": fund.category, "ticker": fund.ticker})
    return rows
