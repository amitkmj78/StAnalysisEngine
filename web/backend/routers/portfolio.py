import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.acquired_at_utils import acquired_at_or_today, is_missing_date
from services.basket_rebalance_service import scan_baskets_for_rebalance
from services.benchmark_comparison_service import compute_benchmark_comparison, compute_benchmark_comparison_multi
from services.data_service import get_effective_price
from services.goal_plan_service import (
    build_signal_weighted_allocation,
    get_annualized_returns,
    solve_goal_plan,
)
from services.index_fund_service import GOAL_DESCRIPTIONS, GOAL_WEIGHTS
from services.manual_positions import build_manual_positions
from services.market_regime_service import regime_as_of
from services.monthly_investing_service import get_best_monthly_pick
from services.portfolio_alert_service import build_drop_analysis, get_price_and_prev_close
from services.portfolio_compare_service import (
    HoldingInput,
    _fetch_close,
    _series_stats,
    build_headline,
    build_portfolio_window_view,
    derive_confidence,
    resolve_window,
    select_gap_drivers,
    select_top_funds,
)
from services.portfolio_health_service import (
    ACCOUNT_TYPES,
    MAX_PARALLEL_HEALTH_FETCHES,
    TOP10_DISCLOSURE,
    build_sector_comparison,
    compute_fee_drag,
    compute_fund_coverage_pct,
    compute_look_through_exposure,
    compute_portfolio_dividend_income,
    compute_portfolio_sector_weights,
    compute_risk_over_windows,
    fetch_fund_holdings_map,
    find_tax_loss_harvest_candidates,
)
from services.portfolio_performance_service import compute_portfolio_performance
from services.portfolio_strategy import build_robinhood_strategies, summarize_portfolio
from services.positions_from_csv import positions_from_activity_csv
from services.ranking_utils import compute_position_concentration
from services.portfolio_review_service import (
    build_portfolio_review,
    compute_market_values,
    compute_sectors,
    flag_positions,
)
from services.sentiment_service import score_tickers_sentiment
from services.signal_publication_service import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PREDICT_DAYS_AHEAD,
    DEFAULT_PREDICT_PERIOD,
    DEFAULT_UNIVERSE,
    compute_predict_algo_comparison,
    rank_within_universe,
)
from services.stock_finder_service import STOCK_UNIVERSES, compute_sp500_sector_mix, rank_stocks_by_window_return
from services.subscriber_events_service import log_event
from services.yfinance_cache import get_cached_dividends, get_cached_info

from web.backend.admin import require_admin
from web.backend.app_settings import (
    PORTFOLIO_DROP_THRESHOLD_DEFAULT,
    PORTFOLIO_DROP_THRESHOLD_PCT_KEY,
    get_setting_float,
)
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
from web.backend.pit_prices import get_signal_stability_for_ticker
from web.backend.llm_cache import cached_init_llms, ordered_llms
from web.backend.portfolio_alerts import scan_portfolios_for_drops
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"], dependencies=[Depends(verify_bearer_token)])

# A position at or above this share of total portfolio value gets flagged
# as concentrated — see portfolio_insights below.
CONCENTRATION_THRESHOLD_PCT = 25.0


def _nan_to_none(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


async def _resolve_portfolio_id(conn, user_id: str, portfolio_id: Optional[int]) -> int:
    """
    A caller can name a specific portfolio (and it must actually belong to
    them and be active — never trust a raw id from the client without
    checking) or omit one entirely, in which case this resolves to their
    oldest active portfolio, auto-creating "My Portfolio" if they don't
    have one yet (a brand-new user's very first save, or every portfolio
    they had has been deactivated by an admin). Every endpoint below goes
    through this rather than trusting portfolio_id directly, so an id for
    someone else's portfolio — or one an admin has deactivated — 404s
    instead of silently reading/writing across accounts.
    """
    if portfolio_id is not None:
        row = await conn.fetchrow(
            "SELECT id FROM portfolios WHERE id = $1 AND user_id = $2::uuid AND is_active",
            portfolio_id, user_id,
        )
        if row is None:
            raise HTTPException(404, "Portfolio not found.")
        return row["id"]

    row = await conn.fetchrow(
        "SELECT id FROM portfolios WHERE user_id = $1::uuid AND is_active ORDER BY created_at ASC LIMIT 1",
        user_id,
    )
    if row is not None:
        return row["id"]

    created = await conn.fetchrow(
        "INSERT INTO portfolios (user_id, name) VALUES ($1::uuid, 'My Portfolio') RETURNING id",
        user_id,
    )
    return created["id"]


async def _merge_with_existing(conn, user_id: str, portfolio_id: int, new_holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines freshly-submitted holdings (a manual-entry save, a CSV import,
    a single-position edit) with whatever the user already has saved,
    keyed by ticker. A ticker present in the new submission is taken
    exactly as submitted there (the user just told us the current truth
    for that position); every other already-saved ticker is preserved
    as-is rather than being wiped out — this is what makes saving
    additive instead of "replace the whole portfolio," matching how
    edit_position already behaved for a single ticker.

    Preserved existing rows carry Ticker/Shares/Avg_Cost/Acquired_At (no
    Current_Price/Name — those aren't in portfolio_positions to begin
    with), which is exactly the shape refresh_portfolio already uses:
    _normalize_holdings_row fetches a live price for them, same as a
    refresh would. Acquired_At specifically matters here: without
    carrying it forward, every save would silently reset an untouched
    position's real acquisition date right along with everything else
    _save_and_respond rewrites.
    """
    new_tickers: set[str] = set()
    if not new_holdings_df.empty and "Ticker" in new_holdings_df.columns:
        new_tickers = {str(t).strip().upper() for t in new_holdings_df["Ticker"]}
        # CSV-derived holdings carry "Net_Shares", not "Shares" (see
        # positions_from_csv.py) — normalize before concatenating with the
        # preserved rows below, which always use "Shares". Left as two
        # differently-named columns, _normalize_holdings_row's
        # `"Shares" in row.index` check would find a Shares column (present
        # only because pandas unions columns across concat) full of NaN for
        # every CSV row, silently discarding real quantities from
        # Net_Shares instead of falling back to it.
        if "Net_Shares" in new_holdings_df.columns and "Shares" not in new_holdings_df.columns:
            new_holdings_df = new_holdings_df.rename(columns={"Net_Shares": "Shares"})

    existing = await conn.fetch(
        "SELECT ticker, shares, avg_cost, acquired_at FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
        user_id, portfolio_id,
    )
    preserved_rows = [
        {"Ticker": r["ticker"], "Shares": r["shares"], "Avg_Cost": r["avg_cost"], "Acquired_At": r["acquired_at"]}
        for r in existing
        if r["ticker"] not in new_tickers
    ]

    # A ticker that IS in the new submission but is also already saved
    # (a manual re-entry, a CSV re-import covering an existing holding)
    # is "taken exactly as submitted" per this function's own contract
    # above — but that must not mean silently resetting its real
    # acquired_at to today just because this particular save didn't
    # happen to carry a date for it. Backfill from the existing row
    # whenever the submission left Acquired_At blank; an explicit date
    # in the submission (fresh CSV buy date, user-entered date, a caller
    # already carrying the right value forward) always wins.
    if not new_holdings_df.empty and "Ticker" in new_holdings_df.columns:
        existing_acquired_at = {r["ticker"]: r["acquired_at"] for r in existing}
        if "Acquired_At" not in new_holdings_df.columns:
            new_holdings_df = new_holdings_df.assign(Acquired_At=None)
        new_holdings_df = new_holdings_df.assign(
            Acquired_At=new_holdings_df.apply(
                lambda r: r["Acquired_At"]
                if not is_missing_date(r["Acquired_At"])
                else existing_acquired_at.get(str(r["Ticker"]).strip().upper()),
                axis=1,
            )
        )

    return pd.concat([pd.DataFrame(preserved_rows), new_holdings_df], ignore_index=True, sort=False)


async def _save_and_respond(conn, user_id: str, portfolio_id: int, holdings_df: pd.DataFrame, risk_profile: str, risk_factor: int, source: str):
    await _invalidate_insights_snapshot(conn, user_id, portfolio_id)
    if holdings_df.empty:
        return {"positions": [], "strategies": [], "summary": summarize_portfolio(pd.DataFrame())}

    strat_df = build_robinhood_strategies(holdings_df, risk_profile=risk_profile, risk_factor=risk_factor)
    if strat_df.empty:
        return {"positions": [], "strategies": [], "summary": summarize_portfolio(strat_df)}

    # holdings_df going in here is already the full merged set for THIS
    # portfolio (see _merge_with_existing) — callers are responsible for
    # combining new data with what's already saved before reaching this
    # point. This delete+reinsert is just how that merged snapshot gets
    # written, not a place that decides what's kept vs. dropped: it must
    # never receive only a subset of the portfolio's positions, or the
    # rest would be lost. Scoped by portfolio_id as well as user_id so
    # saving one portfolio never touches another.
    await conn.execute("DELETE FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2", user_id, portfolio_id)
    await conn.execute("DELETE FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2", user_id, portfolio_id)

    position_rows = []
    for _, row in strat_df.iterrows():
        record = await conn.fetchrow(
            """
            INSERT INTO portfolio_positions (
                user_id, portfolio_id, ticker, name, shares, avg_cost, current_price, unrealized_pnl_pct, source, acquired_at
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            user_id, portfolio_id, row["Ticker"], row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
            _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]), source,
            acquired_at_or_today(row.get("Acquired_At")),
        )
        position_rows.append(_record_to_dict(record))

    strategy_rows = []
    for _, row in strat_df.iterrows():
        record = await conn.fetchrow(
            """
            INSERT INTO portfolio_strategies (
                user_id, portfolio_id, ticker, shares, avg_cost, current_price, unrealized_pnl_pct,
                short_term_plan, long_term_plan, risk_profile, risk_factor
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING *
            """,
            user_id, portfolio_id, row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
            _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]),
            row["Short_Term_Plan"], row["Long_Term_Plan"], row["Risk_Profile"], int(row["Risk_Factor"]),
        )
        strategy_rows.append(_record_to_dict(record))

    # Refresh does not touch the watchlist. Watchlist entries are created only by an explicit user action
    # (the "create watchlist from current price" button), never as a side effect of re-saving holdings.
    return {
        "positions": position_rows,
        "strategies": strategy_rows,
        "summary": summarize_portfolio(strat_df),
        "watchlist_alerts_created": 0,
    }


class CreatePortfolioRequest(BaseModel):
    name: str
    account_type: str = "Taxable"


@router.get("/list")
async def list_portfolios(request: Request):
    # is_active excludes anything an admin has deactivated — same
    # reversible-suspension semantics as a deactivated user account, just
    # scoped to one portfolio. See _resolve_portfolio_id for the matching
    # enforcement on direct-id access.
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            """
            SELECT p.id, p.name, p.created_at, p.margin_balance, p.cash_balance, p.account_type, count(pp.id) AS position_count
            FROM portfolios p
            LEFT JOIN portfolio_positions pp ON pp.portfolio_id = p.id AND pp.user_id = p.user_id
            WHERE p.user_id = $1::uuid AND p.is_active
            GROUP BY p.id, p.name, p.created_at, p.margin_balance, p.cash_balance, p.account_type
            ORDER BY p.created_at ASC
            """,
            user_id,
        )
    return {"portfolios": [_record_to_dict(r) for r in records]}


@router.post("/create")
@limiter.limit("10/minute")
async def create_portfolio(request: Request, body: CreatePortfolioRequest):
    await enforce_daily_quota(request, "portfolio/create")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Portfolio name is required.")
    if len(name) > 100:
        raise HTTPException(422, "Portfolio name must be 100 characters or fewer.")
    if body.account_type not in ACCOUNT_TYPES:
        raise HTTPException(422, f"account_type must be one of {ACCOUNT_TYPES}")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            "INSERT INTO portfolios (user_id, name, account_type) VALUES ($1::uuid, $2, $3) "
            "RETURNING id, name, created_at, margin_balance, cash_balance, account_type",
            user_id, name, body.account_type,
        )
    return {**_record_to_dict(record), "position_count": 0}


@router.delete("/{portfolio_id}")
@limiter.limit("10/minute")
async def delete_portfolio(request: Request, portfolio_id: int):
    """Soft-delete (is_active = false) — the same reversible-suspension
    pattern already used when an admin deactivates a portfolio, just
    user-initiated here. Positions/strategies/history are left in place,
    not dropped; they're just hidden from listings and no longer
    resolvable via portfolio_id. If this was the user's only portfolio,
    the next call that omits portfolio_id auto-creates a fresh
    "My Portfolio" (see _resolve_portfolio_id) rather than erroring."""
    await enforce_daily_quota(request, "portfolio/delete")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET is_active = false WHERE id = $1 AND user_id = $2::uuid AND is_active RETURNING id",
            portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"ok": True}


class SetMarginRequest(BaseModel):
    margin_balance: float


@router.put("/{portfolio_id}/margin")
@limiter.limit("20/minute")
async def set_portfolio_margin(request: Request, portfolio_id: int, body: SetMarginRequest):
    """Records money borrowed from the broker against this portfolio (0 =
    no margin used). A liability the user edits directly, not derived
    from anything else — Net Equity (see /performance) is total market
    value minus this."""
    await enforce_daily_quota(request, "portfolio/margin")
    if body.margin_balance < 0:
        raise HTTPException(422, "Margin balance can't be negative.")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET margin_balance = $1 WHERE id = $2 AND user_id = $3::uuid AND is_active RETURNING id, margin_balance",
            body.margin_balance, portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"id": row["id"], "margin_balance": row["margin_balance"]}


class SetCashRequest(BaseModel):
    cash_balance: float


@router.put("/{portfolio_id}/cash")
@limiter.limit("20/minute")
async def set_portfolio_cash(request: Request, portfolio_id: int, body: SetCashRequest):
    """Records uninvested cash sitting in this portfolio (0 = none). An
    asset the user edits directly, not derived from anything else — Total
    Value and Net Equity (see /summary, /performance) add this on top of
    position market value, but it's deliberately excluded from gain-vs-cost
    and benchmark-comparison return percentages, since idle cash has no
    cost basis and including it would understate the real return on what's
    actually invested."""
    await enforce_daily_quota(request, "portfolio/cash")
    if body.cash_balance < 0:
        raise HTTPException(422, "Cash balance can't be negative.")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET cash_balance = $1 WHERE id = $2 AND user_id = $3::uuid AND is_active RETURNING id, cash_balance",
            body.cash_balance, portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"id": row["id"], "cash_balance": row["cash_balance"]}


class SetAccountTypeRequest(BaseModel):
    account_type: str


@router.put("/{portfolio_id}/account-type")
@limiter.limit("20/minute")
async def set_portfolio_account_type(request: Request, portfolio_id: int, body: SetAccountTypeRequest):
    """HLT-4: tax-loss harvesting is only shown for Taxable accounts --
    this is the one place that's set/edited. User-edited directly, same
    as margin_balance/cash_balance above; this app has no way to infer
    account type from anything else in the schema."""
    await enforce_daily_quota(request, "portfolio/account-type")
    if body.account_type not in ACCOUNT_TYPES:
        raise HTTPException(422, f"account_type must be one of {ACCOUNT_TYPES}")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET account_type = $1 WHERE id = $2 AND user_id = $3::uuid AND is_active RETURNING id, account_type",
            body.account_type, portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"id": row["id"], "account_type": row["account_type"]}


class SaveDiversifiedBasketRequest(BaseModel):
    name: str
    goal: str
    universe: str
    picks_per_sector: int
    max_stocks: Optional[int] = None
    total_amount: float
    fractional_shares: bool = False
    sector_weighting: str = "equal_dollar"
    as_of_date: Optional[date] = None
    holdings: List[dict]
    rebalance_frequency: str = "none"
    drift_threshold_pct: float = 5.0


@router.post("/diversified-basket/save")
@limiter.limit("10/minute")
async def save_diversified_basket(request: Request, body: SaveDiversifiedBasketRequest):
    """
    Saves a "Build a Diversified Basket" preview as a brand-new portfolio
    (DI-07) — always a new portfolio, never merged into an existing one,
    so saving the same basket twice creates two separate portfolios by
    design. Goes through the same build_manual_positions + _save_and_respond
    path every other "save these holdings" flow in this app uses (manual
    entry, CSV import, refresh, move) — a portfolio's Holdings table
    renders from `portfolio_strategies` (built via build_robinhood_
    strategies), not directly from `portfolio_positions`, so an earlier
    version of this endpoint that only inserted into portfolio_positions
    left the new portfolio showing zero holdings on the Portfolio page
    even though the positions existed in the database.
    """
    await enforce_daily_quota(request, "portfolio/diversified-basket/save")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Portfolio name is required.")
    if not body.holdings:
        raise HTTPException(422, "No holdings to save.")
    if body.rebalance_frequency not in ("none", "monthly", "quarterly"):
        raise HTTPException(422, "rebalance_frequency must be one of: none, monthly, quarterly")

    user_id = request.state.user["id"]
    target_weights = {h["Ticker"]: h["Weight_pct"] for h in body.holdings}
    invested = sum(h["Amount"] for h in body.holdings)
    leftover_cash = max(body.total_amount - invested, 0.0)

    holdings_df = build_manual_positions(
        names=[h.get("Name", h["Ticker"]) for h in body.holdings],
        tickers=[h["Ticker"] for h in body.holdings],
        shares=[h["Shares"] for h in body.holdings],
        current_prices=[h["Price"] for h in body.holdings],
        avg_costs=[h["Price"] for h in body.holdings],  # cost basis = price at creation, so gain-vs-cost starts at 0
    )

    async with user_conn(user_id) as conn:
        portfolio = await conn.fetchrow(
            """
            INSERT INTO portfolios (
                user_id, name, cash_balance, basket_universe, basket_goal,
                basket_score_as_of, basket_generation_inputs, basket_target_weights,
                rebalance_frequency, drift_threshold_pct, last_rebalance_checked_at
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, now())
            RETURNING id, name, created_at
            """,
            user_id, name, leftover_cash, body.universe, body.goal, body.as_of_date,
            json.dumps({
                "picks_per_sector": body.picks_per_sector, "max_stocks": body.max_stocks,
                "total_amount": body.total_amount, "fractional_shares": body.fractional_shares,
                "sector_weighting": body.sector_weighting,
            }),
            json.dumps(target_weights), body.rebalance_frequency, body.drift_threshold_pct,
        )
        await _save_and_respond(conn, user_id, portfolio["id"], holdings_df, "Balanced", 5, "DiversifiedBasket")

    await log_event(
        user_id, "diversified_basket_saved", resource=str(portfolio["id"]),
        metadata={
            "universe": body.universe, "goal": body.goal,
            "as_of_date": body.as_of_date.isoformat() if body.as_of_date else None,
            "tickers": list(target_weights.keys()),
        },
    )
    return {"id": portfolio["id"], "name": portfolio["name"], "created_at": portfolio["created_at"]}


def _rebalance_alert_to_dict(record) -> dict:
    row = {k: record[k] for k in record.keys()}
    for col in ("drift_summary", "suggested_swaps", "target_weights"):
        if isinstance(row.get(col), str):
            row[col] = json.loads(row[col])
    return row


@router.get("/rebalance-alerts")
async def list_rebalance_alerts(request: Request):
    """Pending rebalance-check alerts for the current user's Diversified
    Basket portfolios — review-and-act, never auto-applied (see
    services/basket_rebalance_service.py). Unlike /drop-alerts, this is a
    plain per-user endpoint (not admin-gated): every basket owner should
    see their own alerts, not just an admin."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            "SELECT * FROM basket_rebalance_alerts WHERE user_id = $1::uuid AND status = 'pending' ORDER BY created_at DESC",
            user_id,
        )
    return {"alerts": [_rebalance_alert_to_dict(r) for r in records]}


@router.post("/watchlist/from-current-prices")
@limiter.limit("10/minute")
async def watchlist_from_current_prices(
    request: Request,
    portfolio_id: Optional[int] = None,
    pct: float = Query(5.0, gt=0, le=50, description="Alert when a holding moves this many percent up or down from its current price."),
):
    """Explicit user action: take each holding's current market price and create one price-above and one
    price-below alert at +/- pct. Re-running replaces this portfolio's earlier price-watch alerts (their reference
    prices move with the market). Refresh never does this."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        pid = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        tickers = [
            r["ticker"]
            for r in await conn.fetch(
                "SELECT DISTINCT ticker FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 ORDER BY ticker",
                user_id, pid,
            )
        ]

    prices: dict[str, float] = {}
    for ticker in tickers:
        price = await run_in_threadpool(get_effective_price, ticker)
        if price is not None and price > 0:
            prices[ticker] = round(float(price), 2)
    skipped = [t for t in tickers if t not in prices]

    created = 0
    async with user_conn(user_id) as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM watchlist_alerts WHERE user_id = $1::uuid AND portfolio_id = $2 AND source = 'portfolio_price_watch'",
                user_id, pid,
            )
            for ticker, price in prices.items():
                for condition_type, threshold in (
                    ("price_above", round(price * (1 + pct / 100), 2)),
                    ("price_below", round(price * (1 - pct / 100), 2)),
                ):
                    await conn.execute(
                        """
                        INSERT INTO watchlist_alerts (user_id, portfolio_id, ticker, condition_type, threshold, source)
                        VALUES ($1::uuid, $2, $3, $4, $5, 'portfolio_price_watch')
                        """,
                        user_id, pid, ticker, condition_type, threshold,
                    )
                    created += 1
    return {"created": created, "pct": pct, "reference_prices": prices, "skipped": skipped}


@router.post("/rebalance-alerts/{alert_id}/dismiss")
async def dismiss_rebalance_alert(request: Request, alert_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE basket_rebalance_alerts SET status = 'dismissed', seen_at = now(), updated_at = now()
            WHERE id = $1 AND user_id = $2::uuid AND status = 'pending' RETURNING id
            """,
            alert_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Rebalance alert not found.")
    return {"ok": True}


@router.post("/rebalance-alerts/{alert_id}/apply")
@limiter.limit("10/minute")
async def apply_rebalance_alert(request: Request, alert_id: int):
    """
    Executes the alert's suggested swaps: deletes each sold ticker's
    position and inserts its replacement, sized at the alert's fresh
    target weight applied to the portfolio's CURRENT total value (using
    each stock's live price at apply time, not the stale alert-time
    snapshot) — same whole-share/fractional convention the original
    generation used. Then updates the portfolio's stored target weights
    and score-as-of to the alert's fresh values. Never runs
    automatically — only in direct response to this user action.
    """
    await enforce_daily_quota(request, "portfolio/rebalance-alerts/apply")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        alert = await conn.fetchrow(
            "SELECT * FROM basket_rebalance_alerts WHERE id = $1 AND user_id = $2::uuid AND status = 'pending'",
            alert_id, user_id,
        )
        if alert is None:
            raise HTTPException(404, "Rebalance alert not found.")
        portfolio_id = alert["portfolio_id"]

        portfolio_row = await conn.fetchrow(
            "SELECT basket_generation_inputs FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            portfolio_id, user_id,
        )
        generation_inputs = (
            json.loads(portfolio_row["basket_generation_inputs"])
            if isinstance(portfolio_row["basket_generation_inputs"], str)
            else (portfolio_row["basket_generation_inputs"] or {})
        )
        fractional_shares = bool(generation_inputs.get("fractional_shares", False))

        suggested_swaps = (
            json.loads(alert["suggested_swaps"]) if isinstance(alert["suggested_swaps"], str) else alert["suggested_swaps"]
        )
        target_weights = (
            json.loads(alert["target_weights"]) if isinstance(alert["target_weights"], str) else alert["target_weights"]
        )

        held = await conn.fetch(
            "SELECT ticker, shares FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, portfolio_id,
        )
        current_prices: dict[str, Optional[float]] = {}
        for row in held:
            current_prices[row["ticker"]] = await run_in_threadpool(get_effective_price, row["ticker"])
        total_value = sum(
            row["shares"] * current_prices[row["ticker"]]
            for row in held
            if current_prices.get(row["ticker"]) is not None
        )

        applied = 0
        for swap in suggested_swaps:
            sell_ticker = swap.get("sell_ticker")
            buy_ticker = swap.get("buy_ticker")
            if sell_ticker:
                await conn.execute(
                    "DELETE FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3",
                    user_id, portfolio_id, sell_ticker,
                )
            if buy_ticker and total_value:
                price = current_prices.get(buy_ticker) or await run_in_threadpool(get_effective_price, buy_ticker)
                weight_pct = target_weights.get(buy_ticker, 0.0)
                if price and weight_pct:
                    target_amount = total_value * (weight_pct / 100.0)
                    shares = round(target_amount / price, 4) if fractional_shares else float(int(target_amount // price))
                    if shares > 0:
                        await conn.execute(
                            """
                            INSERT INTO portfolio_positions (
                                user_id, portfolio_id, ticker, name, shares, avg_cost, current_price, source, acquired_at
                            ) VALUES ($1::uuid, $2, $3, $3, $4, $5, $5, 'DiversifiedBasketRebalance', current_date)
                            """,
                            user_id, portfolio_id, buy_ticker, shares, price,
                        )
                        applied += 1

        await conn.execute(
            """
            UPDATE portfolios SET basket_target_weights = $1::jsonb, basket_score_as_of = $2,
                                   last_rebalance_checked_at = now()
            WHERE id = $3 AND user_id = $4::uuid
            """,
            json.dumps(target_weights), alert["score_as_of"], portfolio_id, user_id,
        )
        await conn.execute(
            "UPDATE basket_rebalance_alerts SET status = 'applied', applied_at = now(), updated_at = now() WHERE id = $1",
            alert_id,
        )

    await log_event(
        user_id, "rebalance_alert_applied", resource=str(portfolio_id),
        metadata={"alert_id": alert_id, "swaps_applied": applied},
    )
    return {"ok": True, "swaps_applied": applied}


@router.post("/rebalance-alerts/scan-now", dependencies=[Depends(require_admin)])
async def scan_rebalance_alerts_now(rebalance_frequency: Optional[str] = None):
    """Manual ops/testing trigger, mirroring POST /drop-alerts/scan-now."""
    inserted = await scan_baskets_for_rebalance(rebalance_frequency_filter=rebalance_frequency)
    return {"inserted": inserted}


class ManualPositionIn(BaseModel):
    name: str = ""
    ticker: str
    shares: float
    current_price: float
    avg_cost: float
    total_return_pct: Optional[float] = None
    acquired_at: Optional[date] = None


class ManualPositionsRequest(BaseModel):
    positions: List[ManualPositionIn]
    risk_profile: str = "Balanced"
    risk_factor: int = 5
    portfolio_id: Optional[int] = None


@router.post("/manual")
@limiter.limit("10/minute")
async def submit_manual_positions(request: Request, body: ManualPositionsRequest):
    await enforce_daily_quota(request, "portfolio/manual")
    if not body.positions:
        raise HTTPException(422, "At least one position is required")

    user_id = request.state.user["id"]

    holdings_df = build_manual_positions(
        names=[p.name for p in body.positions],
        tickers=[p.ticker for p in body.positions],
        shares=[p.shares for p in body.positions],
        current_prices=[p.current_price for p in body.positions],
        avg_costs=[p.avg_cost for p in body.positions],
        total_returns=[p.total_return_pct for p in body.positions],
        acquired_dates=[p.acquired_at for p in body.positions],
    )
    async with user_conn(user_id) as conn:
        portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        merged_df = await _merge_with_existing(conn, user_id, portfolio_id, holdings_df)
        return await _save_and_respond(conn, user_id, portfolio_id, merged_df, body.risk_profile, body.risk_factor, "Manual")


@router.post("/import-csv")
@limiter.limit("10/minute")
async def import_csv(
    request: Request,
    file: UploadFile = File(...),
    risk_profile: str = "Balanced",
    risk_factor: int = 5,
    portfolio_id: Optional[int] = None,
):
    await enforce_daily_quota(request, "portfolio/import-csv")
    user_id = request.state.user["id"]

    raw = await file.read()
    try:
        holdings_df = await run_in_threadpool(positions_from_activity_csv, io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"Could not process CSV: {exc}")

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        merged_df = await _merge_with_existing(conn, user_id, resolved_portfolio_id, holdings_df)
        return await _save_and_respond(conn, user_id, resolved_portfolio_id, merged_df, risk_profile, risk_factor, "Robinhood")


@router.post("/refresh")
@limiter.limit("10/minute")
async def refresh_portfolio(request: Request, risk_profile: str = "Balanced", risk_factor: int = 5, portfolio_id: Optional[int] = None):
    """Re-fetch current market prices for the user's saved positions and
    recompute strategies from them — no re-entry/re-upload required."""
    await enforce_daily_quota(request, "portfolio/refresh")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost, name, acquired_at FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )
        if not records:
            raise HTTPException(404, "No saved portfolio positions to refresh yet.")

        # Built directly (not via build_manual_positions) so no placeholder
        # Current_Price/Unrealized_PnL_% columns are present — that lets
        # _normalize_holdings_row fetch a live price AND derive PnL from it,
        # instead of PnL getting locked in against a 0/stale price first.
        # Acquired_At carries the real saved date through — a refresh is
        # a re-price, not a re-buy, so it must not reset it to today.
        holdings_df = pd.DataFrame(
            {
                "Ticker": [r["ticker"] for r in records],
                "Name": [r["name"] or r["ticker"] for r in records],
                "Shares": [r["shares"] for r in records],
                "Avg_Cost": [r["avg_cost"] for r in records],
                "Acquired_At": [r["acquired_at"] for r in records],
            }
        )
        return await _save_and_respond(conn, user_id, resolved_portfolio_id, holdings_df, risk_profile, risk_factor, "Refreshed")


class PositionEditRequest(BaseModel):
    shares: float
    avg_cost: float
    name: Optional[str] = None
    risk_profile: str = "Balanced"
    risk_factor: int = 5
    portfolio_id: Optional[int] = None
    acquired_at: Optional[date] = None


@router.put("/positions/{ticker}")
@limiter.limit("20/minute")
async def edit_position(request: Request, ticker: str, body: PositionEditRequest):
    """Add-or-update a single position's shares/avg cost without re-entering
    the whole portfolio — rebuilds the full snapshot through the same save
    path (so strategies and auto-watchlist alerts stay in sync), touching
    only this one ticker's numbers. Upsert: if the ticker isn't already
    saved, this appends it as a new position instead of erroring."""
    await enforce_daily_quota(request, "portfolio/edit-position")
    if body.shares <= 0:
        raise HTTPException(422, "Shares must be positive.")
    if body.avg_cost <= 0:
        raise HTTPException(422, "Avg cost must be positive.")

    ticker = ticker.strip().upper()
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        # An edit changing shares/avg_cost isn't a re-buy: leaving
        # acquired_at unset here lets _merge_with_existing backfill the
        # ticker's real saved date automatically (an explicit
        # body.acquired_at, e.g. the user correcting it, still wins), and
        # only a genuinely new ticker falls through to today.
        new_row = pd.DataFrame(
            [{"Ticker": ticker, "Shares": body.shares, "Avg_Cost": body.avg_cost, "Acquired_At": body.acquired_at}]
        )
        merged_df = await _merge_with_existing(conn, user_id, portfolio_id, new_row)
        return await _save_and_respond(conn, user_id, portfolio_id, merged_df, body.risk_profile, body.risk_factor, "Edited")


async def _delete_position_rows(conn, user_id: str, portfolio_id: int, ticker: str) -> bool:
    """
    Removes one ticker from one portfolio — position, strategy, and any
    portfolio_auto watchlist alerts for it. A position's short/long-term
    plan is computed purely from its own numbers (see
    services/portfolio_strategy.py), never from the rest of the
    portfolio, so deleting one ticker never requires recomputing anyone
    else's — a plain scoped delete is correct here, no _save_and_respond
    round trip needed. Returns whether a position actually existed to delete.
    """
    deleted = await conn.fetchrow(
        "DELETE FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3 RETURNING id",
        user_id, portfolio_id, ticker,
    )
    if deleted is None:
        return False
    await conn.execute(
        "DELETE FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3",
        user_id, portfolio_id, ticker,
    )
    await conn.execute(
        "DELETE FROM watchlist_alerts WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3 AND source = 'portfolio_auto'",
        user_id, portfolio_id, ticker,
    )
    await _invalidate_insights_snapshot(conn, user_id, portfolio_id)
    return True


@router.delete("/positions/{ticker}")
@limiter.limit("20/minute")
async def delete_position(request: Request, ticker: str, portfolio_id: Optional[int] = None):
    """Removes one position entirely — not an edit, an irreversible delete
    (the position itself; the portfolio it lived in is untouched)."""
    await enforce_daily_quota(request, "portfolio/delete-position")
    ticker = ticker.strip().upper()
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        found = await _delete_position_rows(conn, user_id, resolved_portfolio_id, ticker)

    if not found:
        raise HTTPException(404, "Position not found.")
    return {"ok": True}


class MovePositionRequest(BaseModel):
    to_portfolio_id: int
    from_portfolio_id: Optional[int] = None
    risk_profile: str = "Balanced"
    risk_factor: int = 5


@router.post("/positions/{ticker}/move")
@limiter.limit("20/minute")
async def move_position(request: Request, ticker: str, body: MovePositionRequest):
    """Moves one position from one portfolio to another — removed from the
    source, merged into the destination (the destination's own existing
    data for this ticker, if any, is preserved/updated exactly the way a
    normal edit_position upsert would, via the same merge path)."""
    await enforce_daily_quota(request, "portfolio/move-position")
    ticker = ticker.strip().upper()
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        from_id = await _resolve_portfolio_id(conn, user_id, body.from_portfolio_id)
        to_id = await _resolve_portfolio_id(conn, user_id, body.to_portfolio_id)
        if from_id == to_id:
            raise HTTPException(422, "Source and destination portfolios must be different.")

        source_row = await conn.fetchrow(
            "SELECT shares, avg_cost, current_price, acquired_at FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3",
            user_id, from_id, ticker,
        )
        if source_row is None:
            raise HTTPException(404, "Position not found in the source portfolio.")

        await _delete_position_rows(conn, user_id, from_id, ticker)

        # A move between portfolios isn't a re-buy either — carry the
        # original acquired_at over rather than letting it reset to today.
        new_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Shares": source_row["shares"],
                    "Avg_Cost": source_row["avg_cost"],
                    "Current_Price": source_row["current_price"],
                    "Acquired_At": source_row["acquired_at"],
                }
            ]
        )
        merged_df = await _merge_with_existing(conn, user_id, to_id, new_row)
        return await _save_and_respond(conn, user_id, to_id, merged_df, body.risk_profile, body.risk_factor, "Moved")


@router.get("/positions")
async def list_positions(request: Request, portfolio_id: Optional[int] = None):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT * FROM portfolio_positions WHERE portfolio_id = $1 ORDER BY created_at DESC",
            resolved_portfolio_id,
        )
    return {"positions": [_record_to_dict(r) for r in records]}


@router.get("/strategies")
async def list_strategies(request: Request, portfolio_id: Optional[int] = None):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT * FROM portfolio_strategies WHERE portfolio_id = $1 ORDER BY created_at DESC",
            resolved_portfolio_id,
        )
    return {"strategies": [_record_to_dict(r) for r in records]}


@router.get("/summary")
async def portfolio_summary(request: Request, portfolio_id: Optional[int] = None):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT cash_balance FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_portfolio_id, user_id,
        )
        records = await conn.fetch(
            "SELECT * FROM portfolio_strategies WHERE portfolio_id = $1", resolved_portfolio_id
        )

    # Cash is added on top of summarize_portfolio's own total_value (pure
    # market value of positions) here in the router, not inside that
    # function -- deliberately excluded from total_pnl_pct, which stays a
    # position-only weighted return, same reasoning as /performance below.
    cash_balance = portfolio_row["cash_balance"] if portfolio_row else 0.0
    rows = [_record_to_dict(r) for r in records]
    if not rows:
        summary = summarize_portfolio(pd.DataFrame())
        summary["total_value"] += cash_balance
        summary["cash_balance"] = cash_balance
        return {"summary": summary}

    df = pd.DataFrame(rows).rename(
        columns={
            "shares": "Shares",
            "current_price": "Current_Price",
            "unrealized_pnl_pct": "Unrealized_PnL_%",
        }
    )
    summary = summarize_portfolio(df)
    summary["total_value"] += cash_balance
    summary["cash_balance"] = cash_balance
    return {"summary": summary}


_EASTERN = ZoneInfo("America/New_York")


def _eastern_today() -> date:
    """The server runs on UTC system time; a naive date.today() call made
    in the evening (after 8pm ET) mislabels a snapshot with tomorrow's
    date. Insights should be dated by the US trading day they were
    computed for, not the server's UTC calendar day."""
    return datetime.now(_EASTERN).date()


async def _invalidate_insights_snapshot(conn, user_id: str, portfolio_id: int) -> None:
    """Deletes today's cached insights snapshot for this portfolio, if
    one exists, so the next GET /insights or /review recomputes fresh
    instead of silently serving Signal/weight_pct/concentration numbers
    from before this change. Without this, editing/adding/deleting/
    moving a position leaves the cached snapshot (and everything built
    on it, including the Portfolio Review's flagged-position weights)
    stale until the day rolls over or someone remembers to hit Refresh
    — exactly the bug that surfaced when a portfolio was edited (shares
    changed, tickers added/removed) after that day's first insights load."""
    await conn.execute(
        "DELETE FROM portfolio_insights_snapshots WHERE user_id = $1::uuid AND portfolio_id = $2 AND as_of_date = $3",
        user_id, portfolio_id, _eastern_today(),
    )


async def _compute_portfolio_insights(records) -> dict:
    """The actual live computation — same engine /predict and
    /top-performers use. Expensive (a model trained per ticker per
    horizon), which is exactly why callers snapshot the result by date
    instead of re-running this on every page load."""
    tickers = [r["ticker"] for r in records]

    concentration_positions = [
        {"ticker": r["ticker"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records
    ]
    concentration_by_ticker = {
        c["ticker"]: c for c in compute_position_concentration(concentration_positions, CONCENTRATION_THRESHOLD_PCT)
    }

    comparison = await run_in_threadpool(
        compute_predict_algo_comparison, tickers, DEFAULT_PREDICT_PERIOD, DEFAULT_PREDICT_DAYS_AHEAD, [1, 5]
    )
    signal_by_ticker = {c["ticker"]: c for c in comparison}

    # Separate call, not an extra_horizon on the call above: the model's
    # forecast only extends `days_ahead` trading days out, so a 30-day
    # figure needs its own days_ahead=30 run. Kept apart from the call
    # above so Signal (and expected_return_pct's existing 10-day meaning,
    # used elsewhere in the app) stays anchored to DEFAULT_PREDICT_DAYS_AHEAD
    # regardless of this addition.
    comparison_30d = await run_in_threadpool(
        compute_predict_algo_comparison, tickers, DEFAULT_PREDICT_PERIOD, 30, []
    )
    thirty_day_by_ticker = {c["ticker"]: c for c in comparison_30d}

    rank_by_ticker = await run_in_threadpool(rank_within_universe, tickers, DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS)

    positions = []
    for r in records:
        t = r["ticker"]
        sig = signal_by_ticker.get(t, {})
        sig_30d = thirty_day_by_ticker.get(t, {})
        rank = rank_by_ticker.get(t, {})
        conc = concentration_by_ticker.get(t, {})
        positions.append(
            {
                "ticker": t,
                "signal": sig.get("predict_signal"),
                "expected_return_pct": sig.get("predict_expected_return_pct"),
                "target_price": sig.get("predict_target_price"),
                "expected_return_pct_1d": sig.get("predict_expected_return_pct_1d"),
                "target_price_1d": sig.get("predict_target_price_1d"),
                "expected_return_pct_5d": sig.get("predict_expected_return_pct_5d"),
                "target_price_5d": sig.get("predict_target_price_5d"),
                "expected_return_pct_30d": sig_30d.get("predict_expected_return_pct"),
                "target_price_30d": sig_30d.get("predict_target_price"),
                "rank": rank.get("rank"),
                "universe_size": rank.get("universe_size"),
                "trailing_return_pct": rank.get("trailing_return_pct"),
                "weight_pct": conc.get("weight_pct"),
                "concentrated": conc.get("concentrated", False),
            }
        )

    return {
        "positions": positions,
        "concentration_threshold_pct": CONCENTRATION_THRESHOLD_PCT,
        "predict_period": DEFAULT_PREDICT_PERIOD,
        "predict_days_ahead": DEFAULT_PREDICT_DAYS_AHEAD,
        "lookback_days": DEFAULT_LOOKBACK_DAYS,
    }


async def _save_insights_snapshot(conn, user_id: str, portfolio_id: int, as_of: date, result: dict) -> datetime:
    row = await conn.fetchrow(
        """
        INSERT INTO portfolio_insights_snapshots
            (user_id, portfolio_id, as_of_date, positions, concentration_threshold_pct,
             predict_period, predict_days_ahead, lookback_days, updated_at)
        VALUES ($1::uuid, $2, $3, $4::jsonb, $5, $6, $7, $8, now())
        ON CONFLICT (user_id, portfolio_id, as_of_date) DO UPDATE SET
            positions = EXCLUDED.positions,
            concentration_threshold_pct = EXCLUDED.concentration_threshold_pct,
            predict_period = EXCLUDED.predict_period,
            predict_days_ahead = EXCLUDED.predict_days_ahead,
            lookback_days = EXCLUDED.lookback_days,
            updated_at = now()
        RETURNING updated_at
        """,
        user_id, portfolio_id, as_of, json.dumps(result["positions"]),
        result["concentration_threshold_pct"], result["predict_period"],
        result["predict_days_ahead"], result["lookback_days"],
    )
    return row["updated_at"]


async def _get_or_compute_insights(conn, user_id: str, resolved_portfolio_id: int, as_of: date) -> dict:
    """Shared by GET /insights and GET /review: today's snapshot if it
    already exists, else compute-and-save it. Factored out so /review can
    reuse the exact same (expensive, cached-by-day) computation rather
    than triggering its own redundant model run."""
    snapshot = await conn.fetchrow(
        """
        SELECT positions, concentration_threshold_pct, predict_period, predict_days_ahead,
               lookback_days, updated_at
        FROM portfolio_insights_snapshots
        WHERE user_id = $1::uuid AND portfolio_id = $2 AND as_of_date = $3
        """,
        user_id, resolved_portfolio_id, as_of,
    )
    if snapshot is not None:
        return {
            "positions": json.loads(snapshot["positions"]),
            "concentration_threshold_pct": snapshot["concentration_threshold_pct"],
            "predict_period": snapshot["predict_period"],
            "predict_days_ahead": snapshot["predict_days_ahead"],
            "lookback_days": snapshot["lookback_days"],
            "as_of_date": str(as_of),
            "updated_at": snapshot["updated_at"].isoformat(),
        }

    records = await conn.fetch(
        "SELECT ticker, shares, current_price FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
        user_id, resolved_portfolio_id,
    )
    if not records:
        return {"positions": [], "as_of_date": str(as_of), "updated_at": None}

    result = await _compute_portfolio_insights(records)
    updated_at = await _save_insights_snapshot(conn, user_id, resolved_portfolio_id, as_of, result)
    return {**result, "as_of_date": str(as_of), "updated_at": updated_at.isoformat()}


@router.get("/insights")
@limiter.limit("10/minute")
async def portfolio_insights(request: Request, portfolio_id: Optional[int] = None):
    """
    Per-holding live BUY/SELL/HOLD signal + expected return (same engine
    /predict uses) at its native predict_days_ahead horizon (10 trading
    days — expected_return_pct/target_price) plus 1, 5, and 30 trading
    days, momentum rank within the full universe (same rule as
    /top-performers and the published track record), and a concentration
    check — so the page can answer "should I be worried about anything I
    hold," not just what it's worth.

    Computing this is expensive (a model trained per ticker per horizon),
    so the result is snapshotted by US trading day: the first call each
    day computes fresh and saves it; every call after that for the same
    day+portfolio returns the saved snapshot instantly instead of
    re-running the model. Use POST /insights/refresh to force a new
    computation before the day naturally rolls over (e.g. after a
    position changes intraday).
    """
    await enforce_daily_quota(request, "portfolio/insights")
    user_id = request.state.user["id"]
    as_of = _eastern_today()

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        result = await _get_or_compute_insights(conn, user_id, resolved_portfolio_id, as_of)

    return result


@router.post("/insights/refresh")
@limiter.limit("10/minute")
async def portfolio_insights_refresh(request: Request, portfolio_id: Optional[int] = None):
    """Forces a fresh computation regardless of today's existing snapshot
    (e.g. after a position changes intraday) and overwrites it — same
    shape as GET /insights, so the frontend can just swap the response in."""
    await enforce_daily_quota(request, "portfolio/insights/refresh")
    user_id = request.state.user["id"]
    as_of = _eastern_today()

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )
        if not records:
            return {"positions": [], "as_of_date": str(as_of), "updated_at": None}

        result = await _compute_portfolio_insights(records)
        updated_at = await _save_insights_snapshot(conn, user_id, resolved_portfolio_id, as_of, result)

    return {**result, "as_of_date": str(as_of), "updated_at": updated_at.isoformat()}


COMPARE_WINDOWS = ["10D", "30D", "60D", "90D", "1Y"]
COMPARE_SIGNAL_ACTION = {"BUY": "buy", "SELL": "trim", "HOLD": "hold"}


async def _attach_stock_forecasts(stock_rows: list[dict]) -> None:
    """
    Fills in each top_stocks row's signal + expected_return_pct/
    target_price from the already-captured pit_quant_signal snapshot --
    same "one bulk query, not a live per-ticker model run" pattern
    web/backend/routers/entry_strategy.py's _attach_quant_signals already
    uses (that model run is real, expensive compute; this page shows up
    to 10 stocks per load, so re-running it live here would meaningfully
    add to the page's own cost). A ticker missing from today's capture
    (outside the default capture universe, or a capture gap) just gets
    nulls, mutated in place, rather than being dropped from the list.
    Confidence is derived from the same signal-stability data used for
    portfolio holdings (services.portfolio_compare_service.derive_confidence).
    """
    tickers = [r["ticker"] for r in stock_rows]
    if not tickers:
        return

    async with service_conn() as conn:
        as_of_date = await conn.fetchval("SELECT max(as_of_date) FROM pit_quant_signal")
        rows = []
        if as_of_date is not None:
            rows = await conn.fetch(
                """
                SELECT ticker, signal, expected_return_pct, target_price
                FROM pit_quant_signal
                WHERE as_of_date = $1 AND ticker = ANY($2::text[])
                """,
                as_of_date, tickers,
            )
    by_ticker = {r["ticker"]: r for r in rows}

    stability_results = await asyncio.gather(*(get_signal_stability_for_ticker(t) for t in tickers))
    stability_by_ticker = dict(zip(tickers, stability_results))
    # REG-3: today's regime, fetched once for the whole batch rather than
    # per-row -- these are live current-day forecasts, so "today" is the
    # correct regime for every row here (unlike a historical track-record
    # row, which needs its own target_date's regime instead).
    current_regime = await regime_as_of()

    for row in stock_rows:
        q = by_ticker.get(row["ticker"])
        row["expected_return_pct"] = float(q["expected_return_pct"]) if q and q["expected_return_pct"] is not None else None
        row["target_price"] = float(q["target_price"]) if q and q["target_price"] is not None else None
        row["signal"] = {
            "action": COMPARE_SIGNAL_ACTION.get(q["signal"]) if q else None,
            **derive_confidence(stability_by_ticker.get(row["ticker"])),
            "regime": current_regime,
            "as_of": str(as_of_date) if as_of_date else None,
        }


@router.get("/compare")
@limiter.limit("10/minute")
async def portfolio_compare(
    request: Request,
    goal: str = Query(next(iter(GOAL_WEIGHTS))),
    window: str = Query("90D"),
    portfolio_id: Optional[int] = None,
):
    """
    Single read-only call powering the "Portfolio vs. Top Picks" compare
    page — every number on the page comes from this one response, so
    switching goal or window is one request and everything (portfolio
    figures, benchmark, top fund, gap drivers, headline) stays internally
    consistent with everything else on the page. Stage 1: fully
    live-computed on every call, same cost profile as the page's previous
    7-endpoint client-side fan-out (no response caching yet).

    No `overlap_pct` anywhere in this response — that needs real per-
    stock fund constituent weights, which don't exist anywhere in this
    app or its data sources; omitted rather than faked. Each holding's
    `signal.confidence` is derived from existing signal-stability data
    (see services.portfolio_compare_service.derive_confidence), not a
    new model output.
    """
    await enforce_daily_quota(request, "portfolio/compare")
    if goal not in GOAL_WEIGHTS:
        raise HTTPException(422, f"goal must be one of {sorted(GOAL_WEIGHTS)}")
    if window not in COMPARE_WINDOWS:
        raise HTTPException(422, f"window must be one of {COMPARE_WINDOWS}")

    user_id = request.state.user["id"]
    as_of = _eastern_today()

    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT id, name, cash_balance FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_id, user_id,
        )
        records = await conn.fetch(
            "SELECT ticker, shares, current_price, acquired_at FROM portfolio_positions "
            "WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
        insights = await _get_or_compute_insights(conn, user_id, resolved_id, as_of)

    holdings = [HoldingInput(r["ticker"], r["shares"], r["current_price"], r["acquired_at"]) for r in records]

    try:
        bounds = await run_in_threadpool(resolve_window, window)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    portfolio_view = await run_in_threadpool(
        build_portfolio_window_view, holdings, bounds, portfolio_row["cash_balance"] or 0.0
    )

    spy_close = await run_in_threadpool(_fetch_close, "SPY")
    spy_stats = _series_stats(spy_close, bounds.start, bounds.end)
    spy_info = await run_in_threadpool(get_cached_info, "SPY")

    top_funds = await run_in_threadpool(select_top_funds, goal, bounds)
    portfolio_tickers = {h.ticker for h in holdings}
    top_stocks = await run_in_threadpool(rank_stocks_by_window_return, window, "All", 10, portfolio_tickers)
    await _attach_stock_forecasts(top_stocks)

    signal_by_ticker = {p["ticker"]: p for p in insights.get("positions", [])}
    stability_results = await asyncio.gather(*(get_signal_stability_for_ticker(h.ticker) for h in holdings))
    stability_by_ticker = dict(zip((h.ticker for h in holdings), stability_results))
    # REG-3: today's regime for this live holdings view -- see the same
    # note in _attach_stock_forecasts above.
    current_regime = await regime_as_of()

    holdings_out = []
    signal_counts = {"buy": 0, "hold": 0, "trim": 0}
    for row in portfolio_view["holdings"]:
        sig = signal_by_ticker.get(row["ticker"], {})
        action = COMPARE_SIGNAL_ACTION.get(sig.get("signal"))
        if action:
            signal_counts[action] += 1
        holdings_out.append({
            **row,
            "signal": {
                "action": action,
                **derive_confidence(stability_by_ticker.get(row["ticker"])),
                "regime": current_regime,
                "as_of": insights.get("as_of_date"),
            },
        })

    top_fund = top_funds[0] if top_funds else None
    headline = build_headline(
        window, portfolio_view["return_pct"], spy_stats["return_pct"],
        top_fund["ticker"] if top_fund else None, top_fund["return_pct"] if top_fund else None,
    )

    return {
        "as_of": str(as_of),
        "window": {
            "code": bounds.code, "start": bounds.start.date().isoformat(),
            "end": bounds.end.date().isoformat(), "trading_days": bounds.trading_days,
        },
        "goal": {"code": goal, "label": goal, "description": GOAL_DESCRIPTIONS.get(goal)},
        "portfolio": {
            "id": portfolio_row["id"], "name": portfolio_row["name"], "holdings_count": len(holdings),
            "return_pct": portfolio_view["return_pct"], "volatility_pct": portfolio_view["volatility_pct"],
            "max_drawdown_pct": portfolio_view["max_drawdown_pct"], "signals": signal_counts,
            "series": portfolio_view["series"],
        },
        "benchmark": {
            "ticker": "SPY", "return_pct": spy_stats["return_pct"], "volatility_pct": spy_stats["volatility_pct"],
            "expense_ratio_pct": spy_info.get("netExpenseRatio"), "series": spy_stats["series"],
        },
        "top_funds": top_funds,
        "holdings": holdings_out,
        "gap_drivers": select_gap_drivers(portfolio_view["holdings"]),
        "top_stocks": top_stocks,
        "headline": headline,
    }


@router.get("/insights/forecast-1y")
@limiter.limit("20/minute")
async def insights_forecast_1y(request: Request, ticker: str = Query(..., min_length=1)):
    """
    On-demand only — never bundled into /insights. A 252-trading-day
    recursive forecast is far more expensive than the 10d/30d figures
    there (each recursive step compounds the last), and it's unvalidated
    besides (see the docstring on the removed eager version, and the UI's
    own "unvalidated" label) — so a page loading many positions shouldn't
    pay this cost for every one of them just to render.
    """
    await enforce_daily_quota(request, "portfolio/insights/forecast-1y")
    ticker = ticker.strip().upper()

    comparison = await run_in_threadpool(
        compute_predict_algo_comparison, [ticker], DEFAULT_PREDICT_PERIOD, 252, []
    )
    row = comparison[0] if comparison else {}
    return {
        "ticker": ticker,
        "expected_return_pct": row.get("predict_expected_return_pct"),
        "target_price": row.get("predict_target_price"),
    }


@router.get("/sentiment")
@limiter.limit("10/minute")
async def portfolio_sentiment(request: Request, portfolio_id: Optional[int] = None):
    """
    Today's real, LLM-scored news/earnings sentiment (Bullish/Neutral/
    Bearish + one-sentence reasoning) for every ticker held in this
    portfolio — a "current reading" display, NOT a 5-day/10-day forecast.
    See score_ticker_sentiment's docstring for why that distinction is
    load-bearing here (a related predictive-sentiment signal failed
    backtest validation 4 times in this codebase).

    Cached per ticker per US trading day in ticker_sentiment_snapshots —
    shared across every user/portfolio holding that ticker (unlike
    portfolio_insights_snapshots, which is per-user), since the same
    ticker's sentiment reading is identical regardless of who holds it.
    The first request for a given ticker each day computes fresh; every
    request after that, from any user, is a cache hit.
    """
    await enforce_daily_quota(request, "portfolio/sentiment")
    user_id = request.state.user["id"]
    as_of = _eastern_today()

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT DISTINCT ticker FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )
    tickers = [r["ticker"] for r in records]
    if not tickers:
        return {"sentiment": {}, "as_of_date": str(as_of)}

    async with service_conn() as conn:
        cached_rows = await conn.fetch(
            """
            SELECT ticker, label, reasoning FROM ticker_sentiment_snapshots
            WHERE as_of_date = $1 AND ticker = ANY($2::text[])
            """,
            as_of, tickers,
        )
    sentiment_by_ticker = {r["ticker"]: {"label": r["label"], "reasoning": r["reasoning"]} for r in cached_rows}

    missing = [t for t in tickers if t not in sentiment_by_ticker]
    if missing:
        llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
        llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
        if llms:
            fresh = await run_in_threadpool(score_tickers_sentiment, missing, llms)
        else:
            fresh = {t: {"label": None, "reasoning": None} for t in missing}

        # ON CONFLICT DO NOTHING, not DO UPDATE: same immutable-snapshot
        # pattern as pit_quant_signal/pit_analyst_rating (app_service only
        # has INSERT, not UPDATE, on this table — a DO UPDATE clause needs
        # UPDATE privilege too). If two requests race on the same missing
        # ticker, whichever insert lands first wins; harmless for a same-day
        # cache, and this endpoint's response uses its own freshly computed
        # `fresh` values regardless of which row actually persisted.
        async with service_conn() as conn:
            for ticker, result in fresh.items():
                await conn.execute(
                    """
                    INSERT INTO ticker_sentiment_snapshots (ticker, as_of_date, label, reasoning, updated_at)
                    VALUES ($1, $2, $3, $4, now())
                    ON CONFLICT (ticker, as_of_date) DO NOTHING
                    """,
                    ticker, as_of, result["label"], result["reasoning"],
                )
        sentiment_by_ticker.update(fresh)

    return {"sentiment": sentiment_by_ticker, "as_of_date": str(as_of)}


@router.get("/review")
@limiter.limit("10/minute")
async def portfolio_review(request: Request, portfolio_id: Optional[int] = None):
    """
    AI-assisted "what needs a look" review across the whole portfolio.
    Reuses today's already-computed Signal/concentration (portfolio_
    insights_snapshots, via the same helper GET /insights uses — no
    fresh model run) and Sentiment (ticker_sentiment_snapshots, read-only
    here — no fresh LLM sentiment calls; a ticker not yet scored today
    just shows as unavailable rather than triggering its own fill).
    Adds live per-position dollar value and real sector (both fetched
    fresh here, live prices/`.info` are cheap and short-TTL-cached, not
    part of the once-a-day insights snapshot) so concentration flags
    carry real numbers and can catch sector-wide concentration spread
    across several tickers, not just one oversized position.

    Flagging is deterministic Python (SELL signal, concentrated,
    sentiment/signal agreement, sector concentration — see
    services.portfolio_review_service.flag_positions), so it's always
    available even if every LLM provider is down; the one LLM call only
    turns an already-flagged list into a readable paragraph, same
    "explain, don't instruct" boundary as the Quant Signal AI Context.
    """
    await enforce_daily_quota(request, "portfolio/review")
    user_id = request.state.user["id"]
    as_of = _eastern_today()

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        insights = await _get_or_compute_insights(conn, user_id, resolved_portfolio_id, as_of)
        share_records = await conn.fetch(
            "SELECT ticker, shares FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    positions = insights.get("positions") or []
    if not positions:
        return {"summary": None, "flagged": [], "as_of_date": insights.get("as_of_date")}

    tickers = [p["ticker"] for p in positions]
    async with service_conn() as conn:
        sentiment_rows = await conn.fetch(
            "SELECT ticker, label FROM ticker_sentiment_snapshots WHERE as_of_date = $1 AND ticker = ANY($2::text[])",
            as_of, tickers,
        )
    sentiment_by_ticker = {r["ticker"]: r["label"] for r in sentiment_rows}

    shares_input = [{"ticker": r["ticker"], "shares": r["shares"]} for r in share_records]
    market_value_by_ticker = await run_in_threadpool(compute_market_values, shares_input)
    sector_by_ticker = await run_in_threadpool(compute_sectors, tickers)

    enriched = [
        {
            **p,
            "sentiment_label": sentiment_by_ticker.get(p["ticker"]),
            "market_value": market_value_by_ticker.get(p["ticker"]),
            "sector": sector_by_ticker.get(p["ticker"]),
        }
        for p in positions
    ]
    flagged = flag_positions(enriched)

    summary = None
    if flagged:
        llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
        llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
        if llms:
            summary = await run_in_threadpool(build_portfolio_review, llms, flagged)

    return {
        "summary": summary,
        "flagged": [
            {
                "ticker": f["ticker"],
                "signal": f.get("signal"),
                "weight_pct": f.get("weight_pct"),
                "sentiment_label": f.get("sentiment_label"),
                "market_value": f.get("market_value"),
                "sector": f.get("sector"),
                "reasons": f["reasons"],
            }
            for f in flagged
        ],
        "as_of_date": insights.get("as_of_date"),
    }


@router.get("/health/concentration")
@limiter.limit("20/minute")
async def portfolio_health_concentration(request: Request, portfolio_id: Optional[int] = None):
    """HLT-1 (concentration + sector-vs-S&P-500 half; fund overlap/ETF
    look-through is the separate GET /health/overlap): largest positions
    (services.ranking_utils.compute_position_concentration, reused as-is)
    and a sector-weight comparison against compute_sp500_sector_mix's
    market-cap-weighted S&P 500 approximation (reused as-is). Uses each
    position's stored current_price (same convention as GET /compare),
    not a fresh live-price refetch."""
    await enforce_daily_quota(request, "portfolio/health/concentration")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    positions = [
        {"ticker": r["ticker"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]
    if not positions:
        return {"largest_positions": [], "sector_comparison": [], "as_of_date": str(_eastern_today())}

    concentration = compute_position_concentration(positions, threshold_pct=CONCENTRATION_THRESHOLD_PCT)
    largest_positions = sorted(concentration, key=lambda p: p["weight_pct"], reverse=True)[:10]

    tickers = [p["ticker"] for p in positions]
    sector_by_ticker = await run_in_threadpool(compute_sectors, tickers)
    sectored_positions = [{**p, "sector": sector_by_ticker.get(p["ticker"])} for p in positions]
    portfolio_sector_weights = compute_portfolio_sector_weights(sectored_positions)
    sp500_sector_weights = await run_in_threadpool(compute_sp500_sector_mix)
    sector_comparison = build_sector_comparison(portfolio_sector_weights, sp500_sector_weights)

    return {
        "largest_positions": largest_positions,
        "sector_comparison": sector_comparison,
        "as_of_date": str(_eastern_today()),
    }


@router.get("/health/risk")
@limiter.limit("10/minute")
async def portfolio_health_risk(request: Request, portfolio_id: Optional[int] = None):
    """HLT-2: volatility, beta, correlation to SPY, and max drawdown for
    the real portfolio, over both 1-year and 3-year windows -- see
    services.portfolio_health_service.compute_risk_over_windows for the
    computation (each window's real data_start/data_end is included so
    the response states the exact period used, not just the nominal
    1y/3y request)."""
    await enforce_daily_quota(request, "portfolio/health/risk")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    positions = [
        {"ticker": r["ticker"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]
    windows = await run_in_threadpool(compute_risk_over_windows, positions)
    return {"as_of": str(_eastern_today()), "windows": windows}


@router.get("/health/overlap")
@limiter.limit("10/minute")
async def portfolio_health_overlap(request: Request, portfolio_id: Optional[int] = None):
    """HLT-1's fund overlap / ETF look-through: combines a stock held
    directly with its exposure inside any held fund's disclosed top-10
    holdings (services.portfolio_health_service.compute_look_through_
    exposure). Sector weights are recomputed on the look-through-
    expanded ticker set rather than raw positions -- compute_sectors
    already silently excludes ETFs (no `sector` on a fund's .info), so a
    portfolio heavy in SPY would otherwise show an artificially low tech
    weight; this reuses data already fetched here, at no extra network
    cost."""
    await enforce_daily_quota(request, "portfolio/health/overlap")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    positions = [
        {"ticker": r["ticker"], "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]
    if not positions:
        return {
            "combined_exposure": [], "fund_coverage_pct": {}, "sector_comparison": [],
            "disclosure": TOP10_DISCLOSURE, "as_of_date": str(_eastern_today()),
        }

    tickers = [p["ticker"] for p in positions]
    fund_holdings = await run_in_threadpool(fetch_fund_holdings_map, tickers)
    combined_exposure = compute_look_through_exposure(positions, fund_holdings)
    fund_coverage_pct = compute_fund_coverage_pct(fund_holdings)

    expanded_tickers = [row["ticker"] for row in combined_exposure]
    sector_by_ticker = await run_in_threadpool(compute_sectors, expanded_tickers)
    sectored_expanded = [
        {"ticker": row["ticker"], "sector": sector_by_ticker.get(row["ticker"]), "market_value": row["combined_value"]}
        for row in combined_exposure
    ]
    portfolio_sector_weights = compute_portfolio_sector_weights(sectored_expanded)
    sp500_sector_weights = await run_in_threadpool(compute_sp500_sector_mix)
    sector_comparison = build_sector_comparison(portfolio_sector_weights, sp500_sector_weights)

    return {
        "combined_exposure": combined_exposure,
        "fund_coverage_pct": fund_coverage_pct,
        "sector_comparison": sector_comparison,
        "disclosure": TOP10_DISCLOSURE,
        "as_of_date": str(_eastern_today()),
    }


def _safe_get_cached_info(ticker: str) -> dict:
    """get_cached_info doesn't fail open like most of yfinance_cache
    (confirmed live: a rate-limited ticker raised straight through
    compute_portfolio_risk_metrics's own fan-out earlier in this file) --
    wrapped locally here so one bad ticker can't 500 the whole
    income/fees endpoint."""
    try:
        return get_cached_info(ticker) or {}
    except Exception:
        return {}


def _fetch_dividends_and_info(tickers: list[str]) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_HEALTH_FETCHES) as executor:
        dividends_list = list(executor.map(get_cached_dividends, tickers))
        info_list = list(executor.map(_safe_get_cached_info, tickers))
    return dict(zip(tickers, dividends_list)), dict(zip(tickers, info_list))


@router.get("/health/income-fees")
@limiter.limit("10/minute")
async def portfolio_health_income_fees(request: Request, portfolio_id: Optional[int] = None):
    """HLT-3: trailing/projected dividend income and annual fund fee
    drag in dollars, aggregated across real holdings. A position counts
    as a fund for fee-drag purposes using the same fetch_fund_holdings_
    map classification GET /health/overlap already uses (not
    re-detected)."""
    await enforce_daily_quota(request, "portfolio/health/income-fees")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    positions = [
        {"ticker": r["ticker"], "shares": r["shares"] or 0, "market_value": (r["shares"] or 0) * (r["current_price"] or 0)}
        for r in records if r["ticker"]
    ]
    if not positions:
        return {
            "as_of_date": str(_eastern_today()),
            "dividends": {"by_ticker": [], "total_trailing_income": None, "total_projected_income": None},
            "fee_drag": {"by_fund": [], "total_annual_fee_drag_dollars": None},
        }

    tickers = [p["ticker"] for p in positions]
    dividends_by_ticker, info_by_ticker = await run_in_threadpool(_fetch_dividends_and_info, tickers)
    fund_holdings = await run_in_threadpool(fetch_fund_holdings_map, tickers)

    dividends = compute_portfolio_dividend_income(positions, dividends_by_ticker, info_by_ticker)
    fee_drag = compute_fee_drag(positions, fund_holdings, info_by_ticker)

    return {"as_of_date": str(_eastern_today()), "dividends": dividends, "fee_drag": fee_drag}


@router.get("/health/tax-loss-harvesting")
@limiter.limit("20/minute")
async def portfolio_health_tax_loss_harvesting(request: Request, portfolio_id: Optional[int] = None):
    """HLT-4: positions below cost, shown only for Taxable accounts
    (services.portfolio_health_service.find_tax_loss_harvest_candidates).
    Cheapest of the four health endpoints -- pure SQL filter + pure
    Python, no yfinance calls at all."""
    await enforce_daily_quota(request, "portfolio/health/tax-loss-harvesting")
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT account_type FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_id, user_id,
        )
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost, current_price, unrealized_pnl_pct FROM portfolio_positions "
            "WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_id,
        )
    account_type = portfolio_row["account_type"] if portfolio_row else "Taxable"
    positions = [
        {
            "ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"],
            "current_price": r["current_price"], "unrealized_pnl_pct": r["unrealized_pnl_pct"],
        }
        for r in records if r["ticker"]
    ]
    result = find_tax_loss_harvest_candidates(positions, account_type)
    return {"as_of_date": str(_eastern_today()), "account_type": account_type, **result}


@router.get("/goal-plan")
@limiter.limit("10/minute")
async def goal_plan(
    request: Request,
    target_amount: float = Query(..., gt=0),
    target_date: date = Query(...),
    monthly_amount: Optional[float] = Query(None, ge=0),
    portfolio_id: Optional[int] = None,
    compare_universe: Optional[str] = None,
):
    """
    Given this portfolio's current holdings, a target dollar amount, and a
    target date: computes the required monthly contribution to reach it,
    and what percentage of each month's contribution should go to which
    holding — tilted toward whichever holding currently has the strongest
    short-term (10-day) BUY/SELL/HOLD signal, same engine /insights uses.

    The growth projection itself (both for existing holdings and for
    where new contributions are allocated) instead uses each ticker's own
    3-year trailing annualized return, not that short-term signal —
    compounding a multi-year goal off a 10-day forecast would extrapolate
    a short-term wiggle into an absurd annual rate. The two use cases are
    intentionally different numbers: the signal says where new money
    should tilt *this month*, the trailing return says how fast money
    actually grows over years.

    compare_universe (optional, one of the Stock Screener's universe
    keys) adds a best_stock_comparison: the same target/current_value run
    a second time as if today's balance and every future contribution
    went entirely into that universe's #1-ranked stock right now (same
    ranking engine as /stock-finder/rank and the Monthly Investing Plan's
    auto-pick) instead of this portfolio's actual holdings — a "what if I
    just bought the best stock instead" comparison against the plan above.
    """
    await enforce_daily_quota(request, "portfolio/goal-plan")
    user_id = request.state.user["id"]

    today = date.today()
    months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
    if target_date.day < today.day:
        months -= 1
    if months < 1:
        raise HTTPException(422, "target_date must be at least one month in the future.")

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    holdings = [r for r in records if (r["shares"] or 0) > 0 and r["ticker"]]
    if not holdings:
        raise HTTPException(422, "This portfolio has no positions to build a plan around.")

    tickers = [r["ticker"] for r in holdings]
    values = {r["ticker"]: (r["shares"] or 0) * (r["current_price"] or 0) for r in holdings}
    current_value = sum(values.values())

    comparison = await run_in_threadpool(
        compute_predict_algo_comparison, tickers, DEFAULT_PREDICT_PERIOD, DEFAULT_PREDICT_DAYS_AHEAD
    )
    signal_by_ticker = {c["ticker"]: c for c in comparison}

    annualized_returns = await run_in_threadpool(get_annualized_returns, tickers, 3)

    allocation_input = [
        {
            "ticker": t,
            "signal": signal_by_ticker.get(t, {}).get("predict_signal"),
            "expected_return_pct": signal_by_ticker.get(t, {}).get("predict_expected_return_pct"),
            "annualized_return_pct": annualized_returns.get(t),
            "current_value": values.get(t, 0.0),
        }
        for t in tickers
    ]
    allocation = build_signal_weighted_allocation(allocation_input)

    current_blended = (
        sum((values.get(t, 0.0) / current_value) * (annualized_returns.get(t) or 0.0) for t in tickers)
        if current_value > 0
        else None
    )
    contribution_blended = sum(
        (a["weight_pct"] / 100.0) * (a["annualized_return_pct"] or 0.0) for a in allocation
    )

    plan = solve_goal_plan(
        current_value=current_value,
        target_amount=target_amount,
        months=months,
        current_holdings_annualized_return_pct=current_blended,
        contribution_annualized_return_pct=contribution_blended,
        monthly_amount=monthly_amount,
    )

    required_monthly = plan.get("required_monthly_contribution")
    for a in allocation:
        a["monthly_amount"] = round(required_monthly * a["weight_pct"] / 100.0, 2) if required_monthly else 0.0

    missing_return_tickers = [t for t in tickers if annualized_returns.get(t) is None]
    warnings = (
        [f"No multi-year price history for: {', '.join(missing_return_tickers)} — excluded from return blending."]
        if missing_return_tickers
        else []
    )

    best_stock_comparison = None
    if compare_universe is not None:
        if compare_universe not in STOCK_UNIVERSES:
            raise HTTPException(422, f"compare_universe must be one of {sorted(STOCK_UNIVERSES.keys())}")
        # Same Short Term / Long Term split the Stock Screener and Monthly
        # Investing Plan already use — derived from the goal's own horizon
        # rather than asked as a separate input.
        pick_goal = "Short Term" if months <= 12 else "Long Term"
        recommendation = await run_in_threadpool(get_best_monthly_pick, "Stock", pick_goal, compare_universe)
        if recommendation is None or recommendation.expected_return_pct is None:
            warnings.append(f"No ranked pick with enough price history in {compare_universe} for this comparison.")
        else:
            best_plan = solve_goal_plan(
                current_value=current_value,
                target_amount=target_amount,
                months=months,
                current_holdings_annualized_return_pct=recommendation.expected_return_pct,
                contribution_annualized_return_pct=recommendation.expected_return_pct,
                monthly_amount=monthly_amount,
            )
            best_stock_comparison = {
                "ticker": recommendation.ticker,
                "name": recommendation.name,
                "annualized_return_pct": recommendation.expected_return_pct,
                "universe": compare_universe,
                "goal": pick_goal,
                **best_plan,
            }

    return {
        "portfolio_id": resolved_portfolio_id,
        "months_remaining": months,
        "target_amount": target_amount,
        "target_date": target_date.isoformat(),
        "current_value": current_value,
        "current_holdings_annualized_return_pct": current_blended,
        "contribution_annualized_return_pct": contribution_blended,
        "allocation": allocation,
        "best_stock_comparison": best_stock_comparison,
        "warnings": warnings,
        **plan,
    }


class SaveGoalRequest(BaseModel):
    name: str = "Goal"
    target_amount: float
    target_date: date
    monthly_amount: Optional[float] = None
    compare_universe: Optional[str] = None
    portfolio_id: Optional[int] = None


@router.post("/goal-plan/saved")
async def save_goal(request: Request, body: SaveGoalRequest):
    """Saves the inputs to a GET /goal-plan call — not a frozen result.
    Re-running a saved goal (GET /goal-plan with these same params) always
    reflects today's prices/signals/trailing returns, same as re-loading
    the calculator with the same values typed in again."""
    if body.target_amount <= 0:
        raise HTTPException(422, "target_amount must be greater than 0.")
    if body.compare_universe is not None and body.compare_universe not in STOCK_UNIVERSES:
        raise HTTPException(422, f"compare_universe must be one of {sorted(STOCK_UNIVERSES.keys())}")

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, body.portfolio_id)
        record = await conn.fetchrow(
            """
            INSERT INTO saved_portfolio_goals
                (user_id, portfolio_id, name, target_amount, target_date, monthly_amount, compare_universe)
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7)
            RETURNING *
            """,
            user_id, resolved_portfolio_id, body.name.strip() or "Goal",
            body.target_amount, body.target_date, body.monthly_amount, body.compare_universe,
        )
    return {"goal": _record_to_dict(record)}


@router.get("/goal-plan/saved")
async def list_saved_goals(request: Request, portfolio_id: Optional[int] = None):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT * FROM saved_portfolio_goals WHERE user_id = $1::uuid AND portfolio_id = $2 "
            "ORDER BY created_at DESC",
            user_id, resolved_portfolio_id,
        )
    return {"goals": [_record_to_dict(r) for r in records]}


@router.delete("/goal-plan/saved/{goal_id}")
async def delete_saved_goal(request: Request, goal_id: int):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "DELETE FROM saved_portfolio_goals WHERE id = $1 AND user_id = $2::uuid RETURNING id",
            goal_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Saved goal not found.")
    return {"ok": True}


@router.get("/performance")
@limiter.limit("15/minute")
async def portfolio_performance(request: Request, lookback_days: int = 30, portfolio_id: Optional[int] = None):
    """Live portfolio value vs. what the same shares were worth `lookback_days`
    ago — always priced fresh against the market, not the last-saved snapshot."""
    await enforce_daily_quota(request, "portfolio/performance")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT margin_balance, cash_balance FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_portfolio_id, user_id,
        )
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost, acquired_at FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    margin_balance = portfolio_row["margin_balance"] if portfolio_row else 0.0
    cash_balance = portfolio_row["cash_balance"] if portfolio_row else 0.0
    positions = [
        {"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"], "acquired_at": r["acquired_at"]}
        for r in records
    ]
    if not positions:
        return {
            "lookback_days": lookback_days,
            "rows": [],
            "total_value_now": cash_balance,
            "total_value_30d_ago": 0.0,
            "value_diff": 0.0,
            "value_diff_pct": None,
            "total_cost_basis": 0.0,
            "total_gain_vs_cost": 0.0,
            "total_gain_vs_cost_pct": None,
            "total_day_gain": None,
            "total_day_gain_pct": None,
            "margin_balance": margin_balance,
            "cash_balance": cash_balance,
            "net_equity": cash_balance - margin_balance,
        }

    result = await run_in_threadpool(compute_portfolio_performance, positions, lookback_days)
    # Margin/cash are deliberately folded in here, not inside
    # compute_portfolio_performance: both are portfolio-level fields from
    # the `portfolios` table, unrelated to any individual position's own
    # price/gain math that service already owns. Cash is added to the
    # headline total_value_now (and therefore net_equity), but NOT to
    # total_gain_vs_cost/total_gain_vs_cost_pct/value_diff_pct above --
    # those stay position-only, since idle cash has no cost basis and
    # including it would understate the real return on what's actually
    # invested (this also keeps /benchmark's return comparison, which
    # reads total_gain_vs_cost_pct, cash-free without needing its own change).
    result["total_value_now"] += cash_balance
    result["margin_balance"] = margin_balance
    result["cash_balance"] = cash_balance
    result["net_equity"] = result["total_value_now"] - margin_balance
    return result


@router.get("/benchmark")
@limiter.limit("15/minute")
async def portfolio_benchmark_comparison(request: Request, portfolio_id: Optional[int] = None):
    """How this portfolio's total return compares to the S&P 500 (SPY)
    since it was created — see services/benchmark_comparison_service.py
    for what "since created" approximates and why."""
    await enforce_daily_quota(request, "portfolio/benchmark")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT created_at FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_portfolio_id, user_id,
        )
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    empty_response = {
        "benchmark_ticker": "SPY",
        "portfolio_return_pct": None,
        "benchmark_return_pct": None,
        "benchmark_today_pct": None,
        "gap_pct": None,
        "underperforming": False,
        "worst_positions": [],
        "suggestion": None,
    }
    positions = [{"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"]} for r in records]
    if not positions or portfolio_row is None:
        return empty_response

    return await run_in_threadpool(compute_benchmark_comparison, positions, portfolio_row["created_at"])


@router.get("/benchmark-multi")
@limiter.limit("15/minute")
async def portfolio_benchmark_comparison_multi(request: Request, portfolio_id: Optional[int] = None):
    """Same comparison as GET /benchmark, but against both SPY and RSP
    (equal-weight S&P 500) at once (DI-08) — a new, additive endpoint;
    the existing single-benchmark /benchmark is untouched for its other
    callers."""
    await enforce_daily_quota(request, "portfolio/benchmark-multi")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        portfolio_row = await conn.fetchrow(
            "SELECT created_at FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            resolved_portfolio_id, user_id,
        )
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    def _empty_side(ticker: str) -> dict:
        return {
            "benchmark_ticker": ticker, "portfolio_return_pct": None, "benchmark_return_pct": None,
            "benchmark_today_pct": None, "gap_pct": None, "underperforming": False,
            "worst_positions": [], "suggestion": None,
        }

    positions = [{"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"]} for r in records]
    if not positions or portfolio_row is None:
        return {"spy": _empty_side("SPY"), "rsp": _empty_side("RSP")}

    return await run_in_threadpool(compute_benchmark_comparison_multi, positions, portfolio_row["created_at"])


@router.get("/drop-alerts", dependencies=[Depends(require_admin)])
async def list_drop_alerts(request: Request):
    """Same-day drop alerts for the current user's holdings — sentiment/news
    context plus the Predict-page quant signal, synthesized into a
    recommended-action note. Populated by the scan_portfolio_drops
    scheduler job (off by default; an admin opts in via /admin/settings).

    Only shows an alert while its ticker is still held in at least one
    active portfolio with drop alerts enabled — same gate scan_portfolios_
    for_drops uses to decide whether to keep alerting. Without this, an
    alert created before a position was sold or a portfolio was
    deactivated/disabled would otherwise keep showing indefinitely, since
    nothing else in this table ever removes a row (only dismiss sets
    seen_at, which the frontend already filters client-side)."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            """
            SELECT pda.*
            FROM portfolio_drop_alerts pda
            WHERE EXISTS (
                SELECT 1 FROM portfolio_positions pp
                JOIN portfolios p ON p.id = pp.portfolio_id
                WHERE pp.user_id = pda.user_id AND pp.ticker = pda.ticker
                  AND p.is_active AND p.drop_alerts_enabled
            )
            ORDER BY pda.created_at DESC
            """
        )
    return {"alerts": [_record_to_dict(r) for r in records]}


@router.post("/drop-alerts/{alert_id}/dismiss", dependencies=[Depends(require_admin)])
async def dismiss_drop_alert(alert_id: int, request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolio_drop_alerts SET seen_at = now() WHERE id = $1 AND user_id = $2::uuid RETURNING id",
            alert_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Alert not found.")
    return {"ok": True}


@router.post("/drop-alerts/{alert_id}/refresh", dependencies=[Depends(require_admin)])
@limiter.limit("10/minute")
async def refresh_drop_alert(alert_id: int, request: Request):
    """Re-checks price and regenerates the sentiment/quant-signal narrative
    for one already-existing alert, right now. Unlike POST /drop-alerts/
    refresh (which only looks for brand-new drops elsewhere in the
    portfolio), this updates an alert already shown today in place —
    including if the ticker has since recovered above the alert
    threshold, since the point is showing where things actually stand
    now, not preserving the original trigger condition."""
    await enforce_daily_quota(request, "portfolio/drop-alerts/refresh-one")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "SELECT ticker FROM portfolio_drop_alerts WHERE id = $1 AND user_id = $2::uuid",
            alert_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Alert not found.")
    ticker = row["ticker"]

    quote = await run_in_threadpool(get_price_and_prev_close, ticker)
    if quote is None:
        raise HTTPException(422, "Could not fetch current price data for this ticker.")
    pct_change = round((quote["price"] / quote["prev_close"] - 1.0) * 100, 4)
    drop = {"price": quote["price"], "prev_close": quote["prev_close"], "pct_change": pct_change}

    llm_openai, llm_groq, llm_claude, llm_ollama, labels = await run_in_threadpool(cached_init_llms)
    llms = ordered_llms(None, llm_openai, llm_groq, llm_claude, llm_ollama, labels)
    if llms:
        analysis = await run_in_threadpool(build_drop_analysis, llms, ticker, drop)
    else:
        analysis = {
            "sentiment_summary": None,
            "predicted_signal": None,
            "predicted_expected_return_pct": None,
            "predicted_target_price": None,
            "recommended_action": (
                "No LLM provider is configured on the server — showing the raw price move only, "
                "no sentiment/signal synthesis was possible."
            ),
        }

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            UPDATE portfolio_drop_alerts
            SET prev_close = $1, price_at_check = $2, pct_change = $3,
                sentiment_summary = $4, predicted_signal = $5,
                predicted_expected_return_pct = $6, predicted_target_price = $7,
                recommended_action = $8, updated_at = now()
            WHERE id = $9 AND user_id = $10::uuid
            RETURNING *
            """,
            drop["prev_close"], drop["price"], drop["pct_change"],
            analysis["sentiment_summary"], analysis["predicted_signal"],
            analysis["predicted_expected_return_pct"], analysis["predicted_target_price"],
            analysis["recommended_action"], alert_id, user_id,
        )
    if record is None:
        raise HTTPException(404, "Alert not found.")
    return {"alert": _record_to_dict(record)}


@router.post("/drop-alerts/refresh", dependencies=[Depends(require_admin)])
@limiter.limit("5/minute")
async def refresh_drop_alerts(request: Request):
    """User-triggered, on-demand check for new drops in the current user's
    own holdings only — the same analysis the scheduler runs, just without
    waiting for the next tick (or needing the scheduler enabled at all).
    Still respects the once-per-ticker-per-day dedup: a ticker already
    alerted today keeps its existing alert/narrative unchanged — this only
    surfaces genuinely new drops since the last check.

    Uses this user's own drop_alert_threshold_pct if they've set one,
    falling back to the admin-configured global default otherwise — same
    resolution as GET /drop-alerts/threshold."""
    await enforce_daily_quota(request, "portfolio/drop-alerts/refresh")
    user_id = request.state.user["id"]
    inserted = await scan_portfolios_for_drops(user_id=user_id)
    return {"inserted": inserted}


@router.get("/drop-alerts/threshold", dependencies=[Depends(require_admin)])
async def get_drop_alert_threshold(request: Request):
    """The current user's own drop-alert sensitivity, if they've set one —
    falls back to the admin-configured global default (app_settings) when
    they haven't, same default every user effectively had before this was
    configurable per-user."""
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        custom = await conn.fetchval(
            "SELECT drop_alert_threshold_pct FROM users WHERE id = $1::uuid", user_id
        )
    default = await get_setting_float(
        PORTFOLIO_DROP_THRESHOLD_PCT_KEY, default=PORTFOLIO_DROP_THRESHOLD_DEFAULT
    )
    return {
        "threshold_pct": custom if custom is not None else default,
        "is_custom": custom is not None,
        "default_pct": default,
    }


class SetDropAlertThresholdRequest(BaseModel):
    threshold_pct: Optional[float] = None  # None resets to the admin default


@router.post("/drop-alerts/threshold", dependencies=[Depends(require_admin)])
async def set_drop_alert_threshold(request: Request, body: SetDropAlertThresholdRequest):
    """Sets (or, with threshold_pct omitted/null, clears) the current
    user's own drop-alert sensitivity. Cleared means "use the
    admin-configured global default" going forward."""
    if body.threshold_pct is not None and not (0.1 <= body.threshold_pct <= 50):
        raise HTTPException(422, "threshold_pct must be between 0.1 and 50.")
    user_id = request.state.user["id"]
    async with service_conn() as conn:
        await conn.execute(
            "UPDATE users SET drop_alert_threshold_pct = $1 WHERE id = $2::uuid",
            body.threshold_pct, user_id,
        )
    return {"ok": True, "threshold_pct": body.threshold_pct}


@router.get("/drop-alerts/portfolio-enabled", dependencies=[Depends(require_admin)])
async def get_portfolio_drop_alerts_enabled(request: Request, portfolio_id: int):
    """Whether drop alerts are on for one specific portfolio — independent
    per portfolio, unlike the threshold above which is one value for the
    whole account. A user with several portfolios can silence alerts for
    one (e.g. a buy-and-forget retirement account) while keeping them on
    for the rest."""
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "SELECT drop_alerts_enabled FROM portfolios WHERE id = $1 AND user_id = $2::uuid",
            portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"portfolio_id": portfolio_id, "drop_alerts_enabled": row["drop_alerts_enabled"]}


class SetPortfolioDropAlertsEnabledRequest(BaseModel):
    portfolio_id: int
    enabled: bool


@router.post("/drop-alerts/portfolio-enabled", dependencies=[Depends(require_admin)])
async def set_portfolio_drop_alerts_enabled(request: Request, body: SetPortfolioDropAlertsEnabledRequest):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        row = await conn.fetchrow(
            "UPDATE portfolios SET drop_alerts_enabled = $1 WHERE id = $2 AND user_id = $3::uuid RETURNING drop_alerts_enabled",
            body.enabled, body.portfolio_id, user_id,
        )
    if row is None:
        raise HTTPException(404, "Portfolio not found.")
    return {"portfolio_id": body.portfolio_id, "drop_alerts_enabled": row["drop_alerts_enabled"]}


@router.post("/drop-alerts/scan-now", dependencies=[Depends(require_admin)])
async def scan_drop_alerts_now(threshold_pct: Optional[float] = None):
    """Manual trigger for the same scan the scheduler runs every 15
    minutes — for verifying the pipeline, not routine use. Defaults to the
    admin-configured threshold (same as the scheduled job); pass
    threshold_pct to test a different sensitivity for a one-off run
    without changing the saved setting. No enable-gate: unlike publish-now,
    this doesn't start a public/irreversible record, it just checks
    current holdings and (if a drop is found) analyzes and notifies — the
    same thing that would happen on the next tick anyway."""
    if threshold_pct is None:
        threshold_pct = await get_setting_float(
            PORTFOLIO_DROP_THRESHOLD_PCT_KEY, default=PORTFOLIO_DROP_THRESHOLD_DEFAULT
        )
    inserted = await scan_portfolios_for_drops(threshold_pct=threshold_pct)
    return {"inserted": inserted, "threshold_pct": threshold_pct}
