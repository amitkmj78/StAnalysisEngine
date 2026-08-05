from .index_fund_service import INDEX_FUND_UNIVERSE
from .screener_service import INDEX_MAP


def capture_universe_membership() -> list[dict]:
    """
    TR-3 Phase 2: a snapshot of exactly which tickers belong to which
    universe right now, for both stock universes (INDEX_MAP) and fund
    universes (INDEX_FUND_UNIVERSE). No I/O — these are in-code
    definitions, not fetched data — so this just reads them out into rows
    for a caller to persist. Read-only: doesn't write to the DB itself.
    """
    rows = []
    for universe_key, tickers in INDEX_MAP.items():
        for ticker in tickers:
            rows.append({"asset_type": "stock", "universe_key": universe_key, "ticker": ticker})
    for fund in INDEX_FUND_UNIVERSE:
        rows.append({"asset_type": "fund", "universe_key": fund.category, "ticker": fund.ticker})
    return rows
