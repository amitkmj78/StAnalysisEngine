from typing import Dict, List

import yfinance as yf

from .cache_utils import ttl_cache


@ttl_cache(maxsize=256, ttl_seconds=300)
def search_tickers(query: str) -> List[Dict[str, str]]:
    query = query.strip()
    if not query:
        return []

    try:
        quotes = yf.Search(query, max_results=8).quotes
    except Exception:
        return []

    results = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "name": q.get("shortname") or q.get("longname") or symbol,
            "exchange": q.get("exchDisp") or q.get("exchange") or "",
            "type": q.get("typeDisp") or q.get("quoteType") or "",
        })
    return results
