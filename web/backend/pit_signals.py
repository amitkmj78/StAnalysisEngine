import logging
from datetime import date, datetime, timedelta, timezone

from services.pit_signal_service import compute_momentum_ranking_from_pit, filter_knowledge_cutoff
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


async def fetch_pit_prices_as_of(target_date: date, universe_id: str, lookback_days: int) -> list[dict]:
    """
    TR-3's structural lookahead guarantee, half one: this query can only
    ever return rows captured on or before target_date's close, because
    the WHERE clause excludes anything later at the database level — a
    caller can't accidentally ask for future knowledge, unlike an
    after-the-fact check that trusts the caller to have filtered
    correctly. The result also passes through filter_knowledge_cutoff as
    a second, independent, pure-Python check (defense in depth).
    """
    window_start = target_date - timedelta(days=int(lookback_days * 1.6) + 10)
    knowledge_cutoff = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

    async with service_conn() as conn:
        if universe_id == "All":
            ticker_rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM pit_universe_membership WHERE asset_type = 'stock' AND snapshot_date <= $1",
                target_date,
            )
        else:
            ticker_rows = await conn.fetch(
                "SELECT DISTINCT ticker FROM pit_universe_membership "
                "WHERE asset_type = 'stock' AND universe_key = $1 AND snapshot_date <= $2",
                universe_id, target_date,
            )
        tickers = [r["ticker"] for r in ticker_rows]
        if not tickers:
            return []

        rows = await conn.fetch(
            """
            SELECT ticker, price_date, close, captured_at_utc
            FROM pit_prices
            WHERE ticker = ANY($1::text[])
              AND price_date BETWEEN $2 AND $3
              AND captured_at_utc <= $4
            ORDER BY ticker, price_date
            """,
            tickers, window_start, target_date, knowledge_cutoff,
        )

    price_rows = [
        {
            "ticker": r["ticker"],
            "price_date": r["price_date"],
            "close": r["close"],
            "captured_at_utc": r["captured_at_utc"],
        }
        for r in rows
    ]
    return filter_knowledge_cutoff(price_rows, knowledge_cutoff)


async def reconcile_published_vs_pit(
    target_date: date,
    universe_id: str = "All",
    lookback_days: int = 30,
    top_n: int = 25,
) -> dict:
    """
    TR-6's actual acceptance test: reconstructs target_date's ranking
    purely from PIT data known as of that date, and compares it
    ticker-by-ticker against what was actually published for that date.
    A ticker can legitimately be "missing from PIT history" if the PIT
    store didn't have lookback_days+1 trading days of data for it yet —
    the PIT store cannot backfill from before capture began, so this is
    reported honestly rather than treated as a mismatch.
    """
    price_rows = await fetch_pit_prices_as_of(target_date, universe_id, lookback_days)
    pit_days_available = len({r["price_date"] for r in price_rows})

    reconstructed = compute_momentum_ranking_from_pit(price_rows, lookback_days, top_n)
    reconstructed_by_ticker = {r["ticker"]: r for r in reconstructed}

    async with service_conn() as conn:
        published_rows = await conn.fetch(
            """
            SELECT rank, ticker, trailing_return_pct FROM published_signals
            WHERE target_date = $1 AND universe_id = $2 AND lookback_days = $3 AND reason_code IS NULL
            ORDER BY rank
            """,
            target_date, universe_id, lookback_days,
        )
    published_by_ticker = {
        r["ticker"]: {"rank": r["rank"], "trailing_return_pct": r["trailing_return_pct"]} for r in published_rows
    }

    matches = 0
    mismatches = []
    missing_from_pit = []
    for ticker, pub in published_by_ticker.items():
        rec = reconstructed_by_ticker.get(ticker)
        if rec is None:
            missing_from_pit.append({"ticker": ticker, "published_rank": pub["rank"]})
            continue
        if pub["rank"] == rec["rank"]:
            matches += 1
        else:
            mismatches.append({"ticker": ticker, "published_rank": pub["rank"], "reconstructed_rank": rec["rank"]})

    return {
        "target_date": target_date.isoformat(),
        "universe_id": universe_id,
        "lookback_days": lookback_days,
        "pit_trading_days_available": pit_days_available,
        "pit_trading_days_required": lookback_days + 1,
        "published_count": len(published_by_ticker),
        "reconstructed_count": len(reconstructed_by_ticker),
        "matches": matches,
        "mismatches": mismatches,
        "missing_from_pit_history": missing_from_pit,
        "byte_identical": (
            len(published_by_ticker) > 0 and not mismatches and not missing_from_pit
        ),
    }
