"""BRF-1/BRF-2: daily brief email content-builders.

Both briefs are pure content-builders with no delivery logic of their
own -- the scheduler job that calls these hands the result straight to
notification_dispatcher.dispatch_alert (ticker=None, alert_type=
"morning_brief"/"evening_recap"), the same funnel every other alert
already goes through, so quiet hours/digest/channel preference is
honored for free rather than reimplemented here.
"""

import asyncio
from datetime import datetime
from typing import Optional

from starlette.concurrency import run_in_threadpool

from services.benchmark_comparison_service import BENCHMARK_TICKER
from services.data_service import get_effective_price, get_previous_close
from services.email_service import APP_URL
from services.market_regime_service import regime_as_of
from services.notification_dispatcher import EASTERN
from services.portfolio_performance_service import compute_portfolio_performance
from services.sentiment_service import score_tickers_sentiment
from services.stock_detail_service import upcoming_earnings_in_window
from services.yfinance_cache import get_cached_earnings_dates
from web.backend.db import service_conn

MAX_HOLDINGS_PER_DIRECTION = 3
# Caps the morning brief's own LLM cost at 3 calls/user/day (a cache hit
# for any ticker another user already triggered today costs nothing) --
# the same fan-out-per-holding mistake that exhausted Groq's daily quota
# during the SUM-1/SUM-2 EDGAR backfill (see FILING_SUMMARIES_ENABLED_KEY's
# docstring in app_settings.py), applied here before it ships rather than
# discovered the same way again.
MAX_TOP_NEWS_TICKERS = 3
MAX_OVERNIGHT_MOVERS_SHOWN = 3
MORNING_BRIEF_EARNINGS_WINDOW_DAYS = 1


def _fmt_dollars(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _benchmark_today_pct(ticker: str = BENCHMARK_TICKER) -> Optional[float]:
    """Same day-P&L math as a position's own day_gain_pct (previous close
    vs. the current effective price) -- not the heavier since-inception
    compute_benchmark_comparison, which this brief has no use for."""
    price_now = get_effective_price(ticker)
    prev_close = get_previous_close(ticker)
    if price_now is None or prev_close is None or prev_close == 0:
        return None
    return (price_now / prev_close - 1.0) * 100.0


def _tag_rows_with_is_paper(positions: list[dict], performance: dict) -> list[dict]:
    """compute_portfolio_performance returns one row per input position, in
    the same order (every position here has shares > 0, so
    _compute_position_row never returns None for this caller) -- zip by
    index, not by ticker, since the same ticker can legitimately appear in
    both a real and a paper-trading holding for one user."""
    rows = performance["rows"]
    for pos, row in zip(positions, rows):
        row["is_paper"] = bool(pos.get("is_paper"))
    return rows


def _group_day_gain(rows: list[dict]) -> tuple[float, Optional[float]]:
    day_gain = sum(r["day_gain"] for r in rows)
    value_now = sum(r["value_now"] for r in rows if r["value_now"] is not None)
    value_before_today = value_now - day_gain
    pct = (day_gain / value_before_today * 100.0) if value_before_today else None
    return day_gain, pct


def _render_holdings_block(label: str, rows: list[dict]) -> list[str]:
    day_gain, pct = _group_day_gain(rows)
    gainers = sorted((r for r in rows if r["day_gain"] > 0), key=lambda r: r["day_gain"], reverse=True)[:MAX_HOLDINGS_PER_DIRECTION]
    losers = sorted((r for r in rows if r["day_gain"] < 0), key=lambda r: r["day_gain"])[:MAX_HOLDINGS_PER_DIRECTION]

    lines = [f"{label}: {_fmt_dollars(day_gain)} ({_fmt_pct(pct)}) today."]
    if gainers:
        lines.append("  Top contributors:")
        for r in gainers:
            lines.append(f"    {r['ticker']}: {_fmt_dollars(r['day_gain'])} ({_fmt_pct(r['day_gain_pct'])})")
    if losers:
        lines.append("  Top detractors:")
        for r in losers:
            lines.append(f"    {r['ticker']}: {_fmt_dollars(r['day_gain'])} ({_fmt_pct(r['day_gain_pct'])})")
    return lines


def build_evening_recap(positions: list[dict]) -> Optional[dict]:
    """BRF-2: today's portfolio move next to SPY's own today move, plus
    the day's top contributors/detractors by holding. Returns None when
    there are no positions, or when no holding has a day_gain yet (e.g.
    before the first regular-session close, or a weekend/holiday run) --
    an empty brief isn't worth sending.

    Real and paper-trading holdings are always reported as two separate
    totals, never blended into one dollar figure -- a position sourced
    from a linked Alpaca paper account (portfolio_positions.
    alpaca_paper_account_id is set) is practice money, not real money, and
    a recap that silently mixed the two into "your portfolio" would
    misrepresent both.
    """
    if not positions:
        return None

    performance = compute_portfolio_performance(positions)
    rows_with_gain = [r for r in _tag_rows_with_is_paper(positions, performance) if r["day_gain"] is not None]
    if not rows_with_gain:
        return None

    real_rows = [r for r in rows_with_gain if not r["is_paper"]]
    paper_rows = [r for r in rows_with_gain if r["is_paper"]]
    benchmark_today_pct = _benchmark_today_pct()

    # The subject line only collapses to one "portfolio X% today" figure
    # when there's just one kind of holding -- with both present, summing
    # real and paper day-gain into one number would misrepresent both, so
    # the subject says so instead of guessing which one to lead with.
    if real_rows and not paper_rows:
        _, pct = _group_day_gain(real_rows)
        subject = f"Evening recap: portfolio {_fmt_pct(pct)} today"
    elif paper_rows and not real_rows:
        _, pct = _group_day_gain(paper_rows)
        subject = f"Evening recap: paper-trading portfolio {_fmt_pct(pct)} today"
    else:
        subject = "Evening recap: real + paper-trading holdings today"

    lines = []
    if real_rows:
        lines.extend(_render_holdings_block("Your real holdings", real_rows))
        lines.append("")
    if paper_rows:
        lines.extend(_render_holdings_block("Your PAPER-TRADING holdings (practice money, not real)", paper_rows))
        lines.append("")
    lines.append(f"{BENCHMARK_TICKER}: {_fmt_pct(benchmark_today_pct)} today.")
    lines.append("")
    lines.append(f"See {APP_URL}/portfolio for the full breakdown.")

    return {"subject": subject, "text_body": "\n".join(lines)}


def _select_news_tickers(rows_with_gain: list[dict], signal_change_rows: list) -> list[str]:
    """Signal-change tickers first, then the biggest absolute movers --
    the priority order the plan called for, deduped and capped at
    MAX_TOP_NEWS_TICKERS so the real LLM calls stay bounded."""
    movers = sorted(rows_with_gain, key=lambda r: abs(r["day_gain_pct"] or 0.0), reverse=True)
    candidate_order = [r["ticker"] for r in signal_change_rows] + [r["ticker"] for r in movers]
    news_tickers: list[str] = []
    for t in candidate_order:
        if t not in news_tickers:
            news_tickers.append(t)
    return news_tickers[:MAX_TOP_NEWS_TICKERS]


def _format_morning_brief(
    today,
    rows_with_gain: list[dict],
    signal_change_rows: list,
    earnings_today: list[dict],
    regime: Optional[str],
    news_tickers: list[str],
    sentiment_by_ticker: dict[str, dict],
) -> dict:
    """Pure rendering step -- every input is already-resolved data, no DB
    or network calls here, so each of the five sections (overnight
    moves, signal changes, earnings today, market regime, top news) can
    be exercised independently in a test without mocking either.

    Overnight Moves keeps real and paper-trading holdings as separate
    totals (see build_evening_recap's docstring for why) -- the other
    four sections are ticker-level, not value-level, so a signal change
    or an earnings date applies the same regardless of which account
    holds the ticker and isn't split.
    """
    subject = f"Morning brief for {today.isoformat()}"

    lines = ["Overnight Moves:"]
    real_rows = [r for r in rows_with_gain if not r["is_paper"]]
    paper_rows = [r for r in rows_with_gain if r["is_paper"]]
    if real_rows:
        day_gain, pct = _group_day_gain(real_rows)
        lines.append(f"  Real portfolio: {_fmt_dollars(day_gain)} ({_fmt_pct(pct)})")
        for r in sorted(real_rows, key=lambda r: abs(r["day_gain_pct"] or 0.0), reverse=True)[:MAX_OVERNIGHT_MOVERS_SHOWN]:
            lines.append(f"    {r['ticker']}: {_fmt_pct(r['day_gain_pct'])}")
    if paper_rows:
        day_gain, pct = _group_day_gain(paper_rows)
        lines.append(f"  Paper-trading portfolio (practice money, not real): {_fmt_dollars(day_gain)} ({_fmt_pct(pct)})")
        for r in sorted(paper_rows, key=lambda r: abs(r["day_gain_pct"] or 0.0), reverse=True)[:MAX_OVERNIGHT_MOVERS_SHOWN]:
            lines.append(f"    {r['ticker']}: {_fmt_pct(r['day_gain_pct'])}")
    if not real_rows and not paper_rows:
        lines.append("  No price data yet (pre-market or market closed).")
    lines.append("")

    lines.append("Signal Changes:")
    if signal_change_rows:
        for r in signal_change_rows:
            horizon_label = "Short-term" if r["horizon"] == "short" else "Long-term"
            lines.append(f"  {r['ticker']} ({horizon_label}): {r['old_signal']} → {r['new_signal']}")
    else:
        lines.append("  None today.")
    lines.append("")

    lines.append("Earnings Today:")
    if earnings_today:
        for e in earnings_today:
            lines.append(f"  {e['ticker']}: reports {e['market_timing']}")
    else:
        lines.append("  None today.")
    lines.append("")

    lines.append("Market Regime:")
    lines.append(f"  {regime}" if regime else "  No regime reading available yet.")
    lines.append("")

    lines.append("Top News:")
    if news_tickers:
        for t in news_tickers:
            s = sentiment_by_ticker.get(t, {"label": None, "reasoning": None})
            if s["label"]:
                lines.append(f"  {t} ({s['label']}): {s['reasoning']}")
            else:
                lines.append(f"  {t}: no sentiment reading available.")
    else:
        lines.append("  Nothing notable today.")
    lines.append("")

    lines.append(f"See {APP_URL}/portfolio for the full breakdown.")

    return {"subject": subject, "text_body": "\n".join(lines)}


async def build_morning_brief(
    llms: list, user_id: str, positions: list[dict], watchlisted_tickers: list[str]
) -> Optional[dict]:
    """BRF-1: five short sections -- overnight moves, signal changes,
    earnings today, market regime, top news -- meant to read in about 2
    minutes, not a wall of prose. Returns None when the user has no
    positions (a watchlist alone isn't a portfolio brief). All of this
    function's own logic is gathering already-fetched data for
    _format_morning_brief to render; see that function for the actual
    section-by-section formatting.

    "Overnight moves" reuses the same day_gain as BRF-2's evening recap
    -- get_effective_price is already after-hours-aware, so a pre-market
    move is already baked into today's day_gain by the time this runs.
    "Top news" is capped at MAX_TOP_NEWS_TICKERS real LLM calls (a cache
    hit costs nothing) -- see that constant's docstring for why.
    """
    if not positions:
        return None

    today = datetime.now(EASTERN).date()
    owned_tickers = sorted({p["ticker"] for p in positions})
    all_tickers = sorted(set(owned_tickers) | set(watchlisted_tickers))

    performance = await run_in_threadpool(compute_portfolio_performance, positions)
    rows_with_gain = [r for r in _tag_rows_with_is_paper(positions, performance) if r["day_gain"] is not None]

    async with service_conn() as conn:
        signal_change_rows = await conn.fetch(
            """
            SELECT ticker, horizon, old_signal, new_signal FROM signal_change_alerts
            WHERE user_id = $1::uuid AND alert_date = $2
            """,
            user_id, today,
        )

    frames = await asyncio.gather(*[run_in_threadpool(get_cached_earnings_dates, t) for t in all_tickers])
    earnings_today = []
    for ticker, frame in zip(all_tickers, frames):
        upcoming = upcoming_earnings_in_window(frame, as_of=today, window_days=MORNING_BRIEF_EARNINGS_WINDOW_DAYS)
        if upcoming is not None:
            earnings_today.append({"ticker": ticker, **upcoming})

    regime = await regime_as_of()

    news_tickers = _select_news_tickers(rows_with_gain, signal_change_rows)

    sentiment_by_ticker: dict[str, dict] = {}
    if news_tickers:
        async with service_conn() as conn:
            cached_rows = await conn.fetch(
                """
                SELECT ticker, label, reasoning FROM ticker_sentiment_snapshots
                WHERE as_of_date = $1 AND ticker = ANY($2::text[])
                """,
                today, news_tickers,
            )
        sentiment_by_ticker = {r["ticker"]: {"label": r["label"], "reasoning": r["reasoning"]} for r in cached_rows}

        missing = [t for t in news_tickers if t not in sentiment_by_ticker]
        if missing and llms:
            fresh = await run_in_threadpool(score_tickers_sentiment, missing, llms)
            async with service_conn() as conn:
                for ticker, result in fresh.items():
                    await conn.execute(
                        """
                        INSERT INTO ticker_sentiment_snapshots (ticker, as_of_date, label, reasoning, updated_at)
                        VALUES ($1, $2, $3, $4, now())
                        ON CONFLICT (ticker, as_of_date) DO NOTHING
                        """,
                        ticker, today, result["label"], result["reasoning"],
                    )
            sentiment_by_ticker.update(fresh)

    return _format_morning_brief(
        today, rows_with_gain, signal_change_rows, earnings_today, regime, news_tickers,
        sentiment_by_ticker,
    )
