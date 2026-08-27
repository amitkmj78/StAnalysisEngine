import io
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.goal_plan_service import (
    build_signal_weighted_allocation,
    get_annualized_returns,
    solve_goal_plan,
)
from services.manual_positions import build_manual_positions
from services.monthly_investing_service import get_best_monthly_pick
from services.portfolio_alert_service import build_drop_analysis, get_price_and_prev_close
from services.portfolio_performance_service import compute_portfolio_performance
from services.portfolio_strategy import build_robinhood_strategies, summarize_portfolio
from services.positions_from_csv import positions_from_activity_csv
from services.ranking_utils import compute_position_concentration
from services.signal_publication_service import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_PREDICT_DAYS_AHEAD,
    DEFAULT_PREDICT_PERIOD,
    DEFAULT_UNIVERSE,
    compute_predict_algo_comparison,
    rank_within_universe,
)
from services.stock_finder_service import STOCK_UNIVERSES

from web.backend.admin import require_admin
from web.backend.app_settings import (
    PORTFOLIO_DROP_THRESHOLD_DEFAULT,
    PORTFOLIO_DROP_THRESHOLD_PCT_KEY,
    get_setting_float,
)
from web.backend.auth import verify_bearer_token
from web.backend.db import service_conn, user_conn
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

    Preserved existing rows carry only Ticker/Shares/Avg_Cost (no
    Current_Price/Name — those aren't in portfolio_positions to begin
    with), which is exactly the shape refresh_portfolio already uses:
    _normalize_holdings_row fetches a live price for them, same as a
    refresh would.
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
        "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
        user_id, portfolio_id,
    )
    preserved_rows = [
        {"Ticker": r["ticker"], "Shares": r["shares"], "Avg_Cost": r["avg_cost"]}
        for r in existing
        if r["ticker"] not in new_tickers
    ]

    return pd.concat([pd.DataFrame(preserved_rows), new_holdings_df], ignore_index=True, sort=False)


async def _save_and_respond(conn, user_id: str, portfolio_id: int, holdings_df: pd.DataFrame, risk_profile: str, risk_factor: int, source: str):
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
                user_id, portfolio_id, ticker, name, shares, avg_cost, current_price, unrealized_pnl_pct, source
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING *
            """,
            user_id, portfolio_id, row["Ticker"], row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
            _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]), source,
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

    # Auto-populate the watchlist with this snapshot's short-term upside
    # target / protective stop — the "best strategy" numbers already computed
    # above — so alerts exist without the user setting them up by hand.
    # Tagged 'portfolio_auto' so re-saving/refreshing only replaces these,
    # never alerts the user created themselves on the Watchlist page.
    # Scoped by portfolio_id too, so saving portfolio B doesn't wipe out
    # auto-alerts generated from portfolio A's holdings.
    await conn.execute(
        "DELETE FROM watchlist_alerts WHERE user_id = $1::uuid AND portfolio_id = $2 AND source = 'portfolio_auto'",
        user_id, portfolio_id,
    )
    watchlist_alerts_created = 0
    for _, row in strat_df.iterrows():
        for condition_type, threshold in (
            ("price_above", _nan_to_none(row.get("Target_Price"))),
            ("price_below", _nan_to_none(row.get("Stop_Price"))),
        ):
            if threshold is None or threshold <= 0:
                continue
            await conn.execute(
                """
                INSERT INTO watchlist_alerts (user_id, portfolio_id, ticker, condition_type, threshold, source)
                VALUES ($1::uuid, $2, $3, $4, $5, 'portfolio_auto')
                """,
                user_id, portfolio_id, row["Ticker"], condition_type, threshold,
            )
            watchlist_alerts_created += 1

    return {
        "positions": position_rows,
        "strategies": strategy_rows,
        "summary": summarize_portfolio(strat_df),
        "watchlist_alerts_created": watchlist_alerts_created,
    }


class CreatePortfolioRequest(BaseModel):
    name: str


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
            SELECT p.id, p.name, p.created_at, count(pp.id) AS position_count
            FROM portfolios p
            LEFT JOIN portfolio_positions pp ON pp.portfolio_id = p.id AND pp.user_id = p.user_id
            WHERE p.user_id = $1::uuid AND p.is_active
            GROUP BY p.id, p.name, p.created_at
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

    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            "INSERT INTO portfolios (user_id, name) VALUES ($1::uuid, $2) RETURNING id, name, created_at",
            user_id, name,
        )
    return {**_record_to_dict(record), "position_count": 0}


class ManualPositionIn(BaseModel):
    name: str = ""
    ticker: str
    shares: float
    current_price: float
    avg_cost: float
    total_return_pct: Optional[float] = None


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
            "SELECT ticker, shares, avg_cost, name FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )
        if not records:
            raise HTTPException(404, "No saved portfolio positions to refresh yet.")

        # Built directly (not via build_manual_positions) so no placeholder
        # Current_Price/Unrealized_PnL_% columns are present — that lets
        # _normalize_holdings_row fetch a live price AND derive PnL from it,
        # instead of PnL getting locked in against a 0/stale price first.
        holdings_df = pd.DataFrame(
            {
                "Ticker": [r["ticker"] for r in records],
                "Name": [r["name"] or r["ticker"] for r in records],
                "Shares": [r["shares"] for r in records],
                "Avg_Cost": [r["avg_cost"] for r in records],
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
        new_row = pd.DataFrame([{"Ticker": ticker, "Shares": body.shares, "Avg_Cost": body.avg_cost}])
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
            "SELECT shares, avg_cost, current_price FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2 AND ticker = $3",
            user_id, from_id, ticker,
        )
        if source_row is None:
            raise HTTPException(404, "Position not found in the source portfolio.")

        await _delete_position_rows(conn, user_id, from_id, ticker)

        new_row = pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Shares": source_row["shares"],
                    "Avg_Cost": source_row["avg_cost"],
                    "Current_Price": source_row["current_price"],
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
        records = await conn.fetch(
            "SELECT * FROM portfolio_strategies WHERE portfolio_id = $1", resolved_portfolio_id
        )

    rows = [_record_to_dict(r) for r in records]
    if not rows:
        return {"summary": summarize_portfolio(pd.DataFrame())}

    df = pd.DataFrame(rows).rename(
        columns={
            "shares": "Shares",
            "current_price": "Current_Price",
            "unrealized_pnl_pct": "Unrealized_PnL_%",
        }
    )
    return {"summary": summarize_portfolio(df)}


@router.get("/insights")
@limiter.limit("10/minute")
async def portfolio_insights(request: Request, portfolio_id: Optional[int] = None):
    """
    Per-holding live BUY/SELL/HOLD signal + expected return (same engine
    /predict uses) at its native predict_days_ahead horizon (10 trading
    days — expected_return_pct/target_price) plus 1 and 5 trading days
    (expected_return_pct_1d/target_price_1d, expected_return_pct_5d/
    target_price_5d — all read from the same forecast the 10-day figures
    come from, not separate model runs), momentum rank within the full
    universe (same rule as /top-performers and the published track
    record), and a concentration check — so the page can answer "should I
    be worried about anything I hold," not just what it's worth. All
    computed live, nothing persisted: reusing the exact functions already
    called from /predict and /top-performers rather than a new signal path.
    """
    await enforce_daily_quota(request, "portfolio/insights")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        resolved_portfolio_id = await _resolve_portfolio_id(conn, user_id, portfolio_id)
        records = await conn.fetch(
            "SELECT ticker, shares, current_price FROM portfolio_strategies WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )
    if not records:
        return {"positions": []}

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

    rank_by_ticker = await run_in_threadpool(rank_within_universe, tickers, DEFAULT_UNIVERSE, DEFAULT_LOOKBACK_DAYS)

    positions = []
    for r in records:
        t = r["ticker"]
        sig = signal_by_ticker.get(t, {})
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
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid AND portfolio_id = $2",
            user_id, resolved_portfolio_id,
        )

    positions = [{"ticker": r["ticker"], "shares": r["shares"], "avg_cost": r["avg_cost"]} for r in records]
    if not positions:
        return {
            "lookback_days": lookback_days,
            "rows": [],
            "total_value_now": 0.0,
            "total_value_30d_ago": 0.0,
            "value_diff": 0.0,
            "value_diff_pct": None,
            "total_cost_basis": 0.0,
            "total_gain_vs_cost": 0.0,
            "total_gain_vs_cost_pct": None,
        }

    return await run_in_threadpool(compute_portfolio_performance, positions, lookback_days)


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
