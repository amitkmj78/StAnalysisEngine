import logging
from datetime import date

from starlette.concurrency import run_in_threadpool

from services.portfolio_alert_service import DROP_THRESHOLD_PCT, build_drop_analysis, check_for_drop
from web.backend.db import service_conn
from web.backend.llm_cache import cached_init_llms, resolve_llm

logger = logging.getLogger(__name__)


async def scan_portfolios_for_drops(threshold_pct: float = DROP_THRESHOLD_PCT, user_id: str | None = None) -> int:
    """
    Scans every user's portfolio (or just `user_id`'s, for an on-demand
    per-user refresh) for a same-day drop of threshold_pct or more,
    computing the sentiment + quant-signal recommendation once per distinct
    ticker per run (not once per user holding it — multiple users can hold
    the same ticker), and inserts one alert row per affected user. The
    (user_id, ticker, alert_date) unique constraint means a ticker already
    alerted today for a user is skipped entirely, including skipping the
    expensive analysis work for it — this holds regardless of whether the
    scheduler or a user-triggered refresh is what's calling this. Returns
    the number of new alert rows inserted.
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
                "SELECT user_id, ticker FROM portfolio_drop_alerts WHERE alert_date = $1 AND user_id = $2::uuid",
                today, user_id,
            )
        else:
            already_alerted = await conn.fetch(
                "SELECT user_id, ticker FROM portfolio_drop_alerts WHERE alert_date = $1", today
            )

    already_alerted_keys = {(r["user_id"], r["ticker"]) for r in already_alerted}
    pending = [h for h in holdings if (h["user_id"], h["ticker"]) not in already_alerted_keys]
    if not pending:
        return 0

    llm = None
    llm_checked = False
    ticker_analysis: dict[str, dict | None] = {}
    inserted = 0

    async with service_conn() as conn:
        for row in pending:
            user_id, ticker = row["user_id"], row["ticker"]

            if ticker not in ticker_analysis:
                drop = await run_in_threadpool(check_for_drop, ticker, threshold_pct)
                if drop is None:
                    ticker_analysis[ticker] = None
                else:
                    if not llm_checked:
                        llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
                        if labels:
                            llm = resolve_llm(labels[0], llm_openai, llm_groq, llm_claude, llm_ollama)
                        llm_checked = True

                    if llm is not None:
                        analysis = await run_in_threadpool(build_drop_analysis, llm, ticker, drop)
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
                    ticker_analysis[ticker] = {**drop, **analysis}

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

    if inserted:
        analyzed_tickers = len([v for v in ticker_analysis.values() if v is not None])
        logger.info(
            "Portfolio drop alerts: %d new alerts inserted, %d distinct tickers analyzed",
            inserted, analyzed_tickers,
        )
    return inserted
