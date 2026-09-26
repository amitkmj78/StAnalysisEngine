"""
Scheduled rebalance-check engine for "Build a Diversified Basket"
portfolios (approved scope addition alongside DI-01 through DI-09):
periodically re-ranks the basket's original universe/goal, compares each
holding's current dollar weight against its stored target weight, and
flags any ticker that has fallen out of its sector's fresh top-N with a
same-sector replacement suggestion.

This ONLY writes a review-and-act alert (basket_rebalance_alerts) --
never executes a trade itself. Mirrors web/backend/portfolio_alerts.py's
scan/refresh-in-place shape (scan_portfolios_for_drops), but user-facing
rather than admin-gated, since every basket owner should see their own
alerts, not just an admin.
"""

import json
import logging
from datetime import date
from typing import Optional

from starlette.concurrency import run_in_threadpool

from services.data_service import get_effective_price
from services.stock_finder_service import (
    _trim_to_max_stocks,
    assemble_sector_picks,
    get_basket_candidates,
    replace_basket_ticker,
)
from services.subscriber_events_service import log_event
from web.backend.db import service_conn

logger = logging.getLogger(__name__)


async def check_basket_for_rebalance(portfolio_row, positions: list[dict]) -> Optional[dict]:
    """
    portfolio_row: a `portfolios` record with basket_goal, basket_universe,
    basket_generation_inputs (jsonb: picks_per_sector, max_stocks),
    basket_target_weights (jsonb: {ticker: weight_pct}), drift_threshold_pct.
    positions: this portfolio's current [{"ticker", "shares"}, ...].

    Returns None when there's nothing to flag (no meaningful drift, no
    fallen-out holdings) -- the caller must not write an alert in that
    case. Otherwise returns {"drift_summary", "suggested_swaps",
    "target_weights" (fresh), "score_as_of", "max_drift_pct"}.
    """
    goal = portfolio_row["basket_goal"]
    universe = portfolio_row["basket_universe"]
    if not goal or not universe:
        return None

    generation_inputs = json.loads(portfolio_row["basket_generation_inputs"]) if isinstance(
        portfolio_row["basket_generation_inputs"], str
    ) else (portfolio_row["basket_generation_inputs"] or {})
    stored_target_weights = json.loads(portfolio_row["basket_target_weights"]) if isinstance(
        portfolio_row["basket_target_weights"], str
    ) else (portfolio_row["basket_target_weights"] or {})
    picks_per_sector = generation_inputs.get("picks_per_sector", 2)
    max_stocks = generation_inputs.get("max_stocks")
    drift_threshold_pct = portfolio_row["drift_threshold_pct"]

    held_tickers = [p["ticker"] for p in positions]
    if not held_tickers:
        return None

    # Current dollar weight of each held ticker, from a live price -- the
    # same "what would you actually see today" figure the Portfolio page
    # itself shows, not a stale snapshot.
    prices: dict[str, Optional[float]] = {}
    for ticker in set(held_tickers):
        prices[ticker] = await run_in_threadpool(get_effective_price, ticker)

    market_values = {
        p["ticker"]: p["shares"] * prices[p["ticker"]]
        for p in positions
        if prices.get(p["ticker"]) is not None
    }
    total_value = sum(market_values.values())
    if not total_value:
        return None
    current_weights = {t: v / total_value * 100 for t, v in market_values.items()}

    drift_summary = []
    max_drift_pct = 0.0
    for ticker, current_pct in current_weights.items():
        target_pct = stored_target_weights.get(ticker)
        if target_pct is None:
            continue
        drift = current_pct - target_pct
        max_drift_pct = max(max_drift_pct, abs(drift))
        if abs(drift) > drift_threshold_pct:
            drift_summary.append({
                "ticker": ticker, "target_weight_pct": round(target_pct, 2),
                "current_weight_pct": round(current_pct, 2), "drift_pct": round(drift, 2),
            })

    # Re-rank today, using the exact same eligibility/selection/trim
    # algorithm the original generation used, to find what a fresh
    # generation would hold now.
    eligible, _exclusions, score_as_of = await run_in_threadpool(get_basket_candidates, goal, universe)
    fresh_basket, _sector_notes = await run_in_threadpool(assemble_sector_picks, eligible, picks_per_sector)
    if max_stocks and not fresh_basket.empty and len(fresh_basket) > max_stocks:
        fresh_basket, _trim_notes = await run_in_threadpool(
            _trim_to_max_stocks, fresh_basket, max_stocks, "GICS Sector"
        )
    fresh_tickers = set(fresh_basket["Ticker"]) if not fresh_basket.empty else set()

    suggested_swaps = []
    for ticker in held_tickers:
        if ticker in fresh_tickers or ticker not in stored_target_weights:
            continue  # still in the fresh top-N, or not part of this basket's own picks (e.g. user added it manually)
        replacement = replace_basket_ticker(eligible, current_tickers=held_tickers, removed_ticker=ticker)
        suggested_swaps.append({
            "sell_ticker": ticker,
            "buy_ticker": replacement["Ticker"] if replacement else None,
            "buy_name": replacement.get("Name") if replacement else None,
            "reason": "no longer in this sector's top-ranked picks" if replacement else "no longer eligible; no same-sector replacement available",
        })

    fresh_target_weights = {
        t: round(100.0 / len(fresh_tickers), 4) for t in fresh_tickers
    } if fresh_tickers else stored_target_weights

    if not drift_summary and not suggested_swaps:
        return None

    return {
        "drift_summary": drift_summary,
        "suggested_swaps": suggested_swaps,
        "target_weights": fresh_target_weights,
        "score_as_of": score_as_of or None,
        "max_drift_pct": round(max_drift_pct, 2),
    }


async def scan_baskets_for_rebalance(rebalance_frequency_filter: Optional[str] = None) -> int:
    """
    Iterates every active portfolio with rebalance_frequency != 'none'
    (optionally restricted to just 'monthly' or 'quarterly', for the
    scheduler's month-gated call), and for each one due a check, upserts
    a basket_rebalance_alerts row (refresh-in-place on a same-day re-run,
    same dedup contract as portfolio_drop_alerts). Returns the count of
    alert rows inserted or refreshed.
    """
    today = date.today()

    async with service_conn() as conn:
        if rebalance_frequency_filter is not None:
            portfolios = await conn.fetch(
                """
                SELECT id, user_id, basket_goal, basket_universe, basket_generation_inputs,
                       basket_target_weights, drift_threshold_pct
                FROM portfolios
                WHERE is_active AND rebalance_frequency = $1
                """,
                rebalance_frequency_filter,
            )
        else:
            portfolios = await conn.fetch(
                """
                SELECT id, user_id, basket_goal, basket_universe, basket_generation_inputs,
                       basket_target_weights, drift_threshold_pct
                FROM portfolios
                WHERE is_active AND rebalance_frequency != 'none'
                """
            )
        if not portfolios:
            return 0

    changed = 0
    for portfolio_row in portfolios:
        async with service_conn() as conn:
            positions = await conn.fetch(
                "SELECT ticker, shares FROM portfolio_positions WHERE portfolio_id = $1", portfolio_row["id"]
            )
        positions_list = [{"ticker": p["ticker"], "shares": p["shares"]} for p in positions if p["ticker"]]

        result = await check_basket_for_rebalance(portfolio_row, positions_list)
        if result is None:
            continue

        async with service_conn() as conn:
            outcome = await conn.execute(
                """
                INSERT INTO basket_rebalance_alerts (
                    user_id, portfolio_id, check_date, score_as_of, drift_summary,
                    suggested_swaps, target_weights, max_drift_pct
                ) VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8)
                ON CONFLICT (portfolio_id, check_date) DO UPDATE SET
                    score_as_of = excluded.score_as_of, drift_summary = excluded.drift_summary,
                    suggested_swaps = excluded.suggested_swaps, target_weights = excluded.target_weights,
                    max_drift_pct = excluded.max_drift_pct, status = 'pending', updated_at = now()
                """,
                portfolio_row["user_id"], portfolio_row["id"], today, result["score_as_of"],
                json.dumps(result["drift_summary"]), json.dumps(result["suggested_swaps"]),
                json.dumps(result["target_weights"]), result["max_drift_pct"],
            )
            await conn.execute(
                "UPDATE portfolios SET last_rebalance_checked_at = now() WHERE id = $1", portfolio_row["id"]
            )
        if outcome in ("INSERT 0 1", "UPDATE 1"):
            changed += 1

        await log_event(
            str(portfolio_row["user_id"]), "rebalance_alert_generated", resource=str(portfolio_row["id"]),
            metadata={
                "max_drift_pct": result["max_drift_pct"], "drifted_holdings": len(result["drift_summary"]),
                "suggested_swaps": len(result["suggested_swaps"]),
            },
        )

    if changed:
        logger.info("Basket rebalance checks: %d alerts inserted/refreshed", changed)
    return changed
