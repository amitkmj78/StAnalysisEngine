from datetime import datetime, timezone


def filter_knowledge_cutoff(price_rows: list[dict], knowledge_cutoff: datetime) -> list[dict]:
    """
    TR-3's non-bypassable lookahead guard, as application-code enforcement
    — applied in addition to (not instead of) the SQL WHERE clause the
    caller should already be using to fetch only rows with
    captured_at_utc <= knowledge_cutoff. Belt and suspenders: even if the
    query layer had a bug, no row whose knowledge date is later than the
    cutoff survives this filter. Every row must carry a "captured_at_utc"
    key (a datetime) — this is what makes a row's presence here a hard
    guarantee, not a request.
    """
    cutoff = knowledge_cutoff if knowledge_cutoff.tzinfo else knowledge_cutoff.replace(tzinfo=timezone.utc)
    safe = []
    for row in price_rows:
        captured_at = row["captured_at_utc"]
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=timezone.utc)
        if captured_at <= cutoff:
            safe.append(row)
    return safe


def compute_momentum_ranking_from_pit(
    price_rows: list[dict],
    lookback_days: int,
    top_n: int,
) -> list[dict]:
    """
    The exact same rule as signal_publication_service.build_daily_signal_set
    (trailing lookback_days-trading-day return, sorted descending, top_n) —
    just computed from a pre-fetched list of PIT price rows
    ({"ticker", "price_date", "close"}) instead of a live yfinance fetch.
    This is the data-source swap TR-6 requires: same ranking rule, PIT
    feed instead of a live one.

    A ticker with fewer than lookback_days + 1 PIT price rows is skipped
    outright rather than guessed at — the PIT store cannot backfill
    history from before capture began, so "not enough data yet" is an
    honest, expected outcome for any window that reaches before the
    capture start date, not a bug.
    """
    by_ticker: dict[str, list[tuple]] = {}
    for row in price_rows:
        by_ticker.setdefault(row["ticker"], []).append((row["price_date"], row["close"]))

    scored = []
    for ticker, points in by_ticker.items():
        points.sort(key=lambda p: p[0])
        if len(points) < lookback_days + 1:
            continue
        start_close = points[-lookback_days - 1][1]
        end_close = points[-1][1]
        if not start_close:
            continue
        trailing_return_pct = (float(end_close) / float(start_close) - 1.0) * 100
        scored.append({"ticker": ticker, "trailing_return_pct": round(trailing_return_pct, 4)})

    scored.sort(key=lambda r: r["trailing_return_pct"], reverse=True)
    ranked = scored[:top_n]
    return [{"rank": i + 1, **r} for i, r in enumerate(ranked)]
