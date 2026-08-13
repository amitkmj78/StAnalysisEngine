import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.portfolio_alert_service import (
    DROP_THRESHOLD_PCT,
    build_drop_analysis,
    check_for_drop,
    get_price_and_prev_close,
)
from web.backend.db import service_conn
from web.backend.llm_cache import cached_init_llms, resolve_llm

logger = logging.getLogger(__name__)


async def scan_portfolios_for_drops(threshold_pct: float = DROP_THRESHOLD_PCT, user_id: str | None = None) -> int:
    """
    Scans every user's portfolio (or just `user_id`'s, for an on-demand
    per-user refresh) for a same-day drop of threshold_pct or more,
    computing the sentiment + quant-signal recommendation once per distinct
    ticker per run (not once per user holding it — multiple users can hold
    the same ticker), and inserts one alert row per newly-affected user.

    A ticker already alerted today for a user is NOT skipped — its alert
    is refreshed in place instead (same price re-check + narrative
    regeneration as the single-alert manual refresh endpoint), so a
    holding that keeps dropping through the day doesn't show a stale
    snapshot from whenever it first crossed the threshold. This runs
    every PORTFOLIO_DROP_INTERVAL_MINUTES (15 min) via the scheduler, so
    in practice no alert goes more than ~15 minutes without a refresh.

    Returns the number of *new* alert rows inserted — refreshes of
    existing rows aren't counted here (existing behavior/contract for
    callers that log "N new alerts"), but do happen as a side effect.
    """
    today = date.today()

    async with service_conn() as conn:
        if user_id is not None:
            holdings = await conn.fetch(
                "SELECT DISTINCT user_id, ticker FROM portfolio_positions WHERE ticker IS NOT NULL AND user_id = $1::uuid",
                user_id,
            )
        else:
            holdings = await conn.fetch(
                "SELECT DISTINCT user_id, ticker FROM portfolio_positions WHERE ticker IS NOT NULL"
            )
        if not holdings:
            return 0
        if user_id is not None:
            already_alerted = await conn.fetch(
                "SELECT id, user_id, ticker FROM portfolio_drop_alerts WHERE alert_date = $1 AND user_id = $2::uuid",
                today, user_id,
            )
        else:
            already_alerted = await conn.fetch(
                "SELECT id, user_id, ticker FROM portfolio_drop_alerts WHERE alert_date = $1", today
            )

    already_alerted_by_key = {(r["user_id"], r["ticker"]): r["id"] for r in already_alerted}
    pending = [h for h in holdings if (h["user_id"], h["ticker"]) not in already_alerted_by_key]
    to_refresh = [h for h in holdings if (h["user_id"], h["ticker"]) in already_alerted_by_key]

    llm = None
    llm_checked = False

    async def _get_llm():
        nonlocal llm, llm_checked
        if not llm_checked:
            llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
            if labels:
                llm = resolve_llm(labels[0], llm_openai, llm_groq, llm_claude, llm_ollama)
            llm_checked = True
        return llm

    async def _analyze(ticker: str, drop: dict) -> dict:
        current_llm = await _get_llm()
        if current_llm is not None:
            analysis = await run_in_threadpool(build_drop_analysis, current_llm, ticker, drop)
        else:
            analysis = {
                "sentiment_summary": None,
                "predicted_signal": None,
                "predicted_expected_return_pct": None,
                "predicted_target_price": None,
                "recommended_action": (
                    "No LLM provider is configured on the server — showing the raw price "
                    "move only, no sentiment/signal synthesis was possible."
                ),
            }
        return {**drop, **analysis}

    inserted = 0
    refreshed = 0
    ticker_analysis: dict[str, dict | None] = {}

    async with service_conn() as conn:
        for row in pending:
            user_id, ticker = row["user_id"], row["ticker"]

            if ticker not in ticker_analysis:
                drop = await run_in_threadpool(check_for_drop, ticker, threshold_pct)
                ticker_analysis[ticker] = await _analyze(ticker, drop) if drop is not None else None

            data = ticker_analysis[ticker]
            if data is None:
                continue

            result = await conn.execute(
                """
                INSERT INTO portfolio_drop_alerts (
                    user_id, ticker, alert_date, prev_close, price_at_check, pct_change,
                    sentiment_summary, predicted_signal, predicted_expected_return_pct,
                    predicted_target_price, recommended_action
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (user_id, ticker, alert_date) DO NOTHING
                """,
                user_id, ticker, today, data["prev_close"], data["price"], data["pct_change"],
                data["sentiment_summary"], data["predicted_signal"], data["predicted_expected_return_pct"],
                data["predicted_target_price"], data["recommended_action"],
            )
            if result == "INSERT 0 1":
                inserted += 1

        refresh_analysis: dict[str, dict | None] = {}
        for row in to_refresh:
            user_id, ticker = row["user_id"], row["ticker"]
            alert_id = already_alerted_by_key[(user_id, ticker)]

            if ticker not in refresh_analysis:
                quote = await run_in_threadpool(get_price_and_prev_close, ticker)
                if quote is None:
                    refresh_analysis[ticker] = None
                else:
                    pct_change = round((quote["price"] / quote["prev_close"] - 1.0) * 100, 4)
                    drop = {"price": quote["price"], "prev_close": quote["prev_close"], "pct_change": pct_change}
                    refresh_analysis[ticker] = await _analyze(ticker, drop)

            data = refresh_analysis[ticker]
            if data is None:
                continue

            result = await conn.execute(
                """
                UPDATE portfolio_drop_alerts
                SET prev_close = $1, price_at_check = $2, pct_change = $3,
                    sentiment_summary = $4, predicted_signal = $5,
                    predicted_expected_return_pct = $6, predicted_target_price = $7,
                    recommended_action = $8, updated_at = now()
                WHERE id = $9
                """,
                data["prev_close"], data["price"], data["pct_change"],
                data["sentiment_summary"], data["predicted_signal"], data["predicted_expected_return_pct"],
                data["predicted_target_price"], data["recommended_action"], alert_id,
            )
            if result == "UPDATE 1":
                refreshed += 1

    if inserted or refreshed:
        logger.info(
            "Portfolio drop alerts: %d new alerts inserted, %d existing alerts refreshed",
            inserted, refreshed,
        )
    return inserted
