"""
Pure, zero-dependency ranking helpers. Kept dependency-free on purpose so
they're importable (and testable) without pulling in anything that needs
network access or an API key — services/signal_publication_service.py
(which has a heavy transitive OPENAI_API_KEY import chain) and
web/backend/routers/portfolio.py both call into this module rather than
duplicating the math.
"""


def rank_tickers_against_universe(
    universe_rows: list[dict],
    held_tickers: list[str],
) -> dict[str, dict]:
    """
    Given a precomputed universe table (rows of {"ticker", "trailing_return_pct"},
    already fetched/computed by the caller) and a set of tickers to look up,
    returns each held ticker's 1-indexed rank (1 = best trailing return),
    the universe size, and its own trailing_return_pct. A held ticker
    absent from universe_rows (delisted, not covered, missing data) gets
    rank=None rather than a guess.
    """
    ranked = sorted(
        (r for r in universe_rows if r.get("trailing_return_pct") is not None),
        key=lambda r: r["trailing_return_pct"],
        reverse=True,
    )
    universe_size = len(ranked)
    rank_by_ticker = {r["ticker"]: i + 1 for i, r in enumerate(ranked)}
    return_by_ticker = {r["ticker"]: r["trailing_return_pct"] for r in ranked}

    return {
        t: {
            "rank": rank_by_ticker.get(t),
            "universe_size": universe_size,
            "trailing_return_pct": return_by_ticker.get(t),
        }
        for t in held_tickers
    }


def compute_position_concentration(
    positions: list[dict],
    threshold_pct: float = 25.0,
) -> list[dict]:
    """
    Each position's share of total portfolio market value — the simplest,
    most reliable diversification signal available without new external
    data (sector data would need a fresh per-ticker yfinance .info fetch,
    slower and flakier than anything already loaded for the portfolio
    page). positions: [{"ticker", "market_value"}]. Flags any position at
    or above threshold_pct as concentrated.
    """
    total = sum(p["market_value"] for p in positions)
    if total <= 0:
        return [{"ticker": p["ticker"], "weight_pct": 0.0, "concentrated": False} for p in positions]

    result = []
    for p in positions:
        weight_pct = p["market_value"] / total * 100.0
        result.append(
            {
                "ticker": p["ticker"],
                "weight_pct": round(weight_pct, 2),
                "concentrated": weight_pct >= threshold_pct,
            }
        )
    return result
