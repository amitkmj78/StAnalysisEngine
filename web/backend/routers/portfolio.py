import io
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.manual_positions import build_manual_positions
from services.portfolio_performance_service import compute_portfolio_performance
from services.portfolio_strategy import build_robinhood_strategies, summarize_portfolio
from services.positions_from_csv import positions_from_activity_csv

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"], dependencies=[Depends(verify_bearer_token)])


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


async def _save_and_respond(conn, user_id: str, holdings_df: pd.DataFrame, risk_profile: str, risk_factor: int, source: str):
    if holdings_df.empty:
        return {"positions": [], "strategies": [], "summary": summarize_portfolio(pd.DataFrame())}

    strat_df = build_robinhood_strategies(holdings_df, risk_profile=risk_profile, risk_factor=risk_factor)
    if strat_df.empty:
        return {"positions": [], "strategies": [], "summary": summarize_portfolio(strat_df)}

    # A save always represents the user's current portfolio snapshot, not an
    # addition to history — replace what's there instead of accumulating duplicates.
    await conn.execute("DELETE FROM portfolio_positions WHERE user_id = $1::uuid", user_id)
    await conn.execute("DELETE FROM portfolio_strategies WHERE user_id = $1::uuid", user_id)

    position_rows = []
    for _, row in strat_df.iterrows():
        record = await conn.fetchrow(
            """
            INSERT INTO portfolio_positions (
                user_id, ticker, name, shares, avg_cost, current_price, unrealized_pnl_pct, source
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
            """,
            user_id, row["Ticker"], row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
            _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]), source,
        )
        position_rows.append(_record_to_dict(record))

    strategy_rows = []
    for _, row in strat_df.iterrows():
        record = await conn.fetchrow(
            """
            INSERT INTO portfolio_strategies (
                user_id, ticker, shares, avg_cost, current_price, unrealized_pnl_pct,
                short_term_plan, long_term_plan, risk_profile, risk_factor
            ) VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
            """,
            user_id, row["Ticker"], _nan_to_none(row["Shares"]), _nan_to_none(row["Avg_Cost"]),
            _nan_to_none(row["Current_Price"]), _nan_to_none(row["Unrealized_PnL_%"]),
            row["Short_Term_Plan"], row["Long_Term_Plan"], row["Risk_Profile"], int(row["Risk_Factor"]),
        )
        strategy_rows.append(_record_to_dict(record))

    # Auto-populate the watchlist with this snapshot's short-term upside
    # target / protective stop — the "best strategy" numbers already computed
    # above — so alerts exist without the user setting them up by hand.
    # Tagged 'portfolio_auto' so re-saving/refreshing only replaces these,
    # never alerts the user created themselves on the Watchlist page.
    await conn.execute(
        "DELETE FROM watchlist_alerts WHERE user_id = $1::uuid AND source = 'portfolio_auto'",
        user_id,
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
                INSERT INTO watchlist_alerts (user_id, ticker, condition_type, threshold, source)
                VALUES ($1::uuid, $2, $3, $4, 'portfolio_auto')
                """,
                user_id, row["Ticker"], condition_type, threshold,
            )
            watchlist_alerts_created += 1

    return {
        "positions": position_rows,
        "strategies": strategy_rows,
        "summary": summarize_portfolio(strat_df),
        "watchlist_alerts_created": watchlist_alerts_created,
    }


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
        return await _save_and_respond(conn, user_id, holdings_df, body.risk_profile, body.risk_factor, "Manual")


@router.post("/import-csv")
@limiter.limit("10/minute")
async def import_csv(
    request: Request,
    file: UploadFile = File(...),
    risk_profile: str = "Balanced",
    risk_factor: int = 5,
):
    await enforce_daily_quota(request, "portfolio/import-csv")
    user_id = request.state.user["id"]

    raw = await file.read()
    try:
        holdings_df = await run_in_threadpool(positions_from_activity_csv, io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(422, f"Could not process CSV: {exc}")

    async with user_conn(user_id) as conn:
        return await _save_and_respond(conn, user_id, holdings_df, risk_profile, risk_factor, "Robinhood")


@router.post("/refresh")
@limiter.limit("10/minute")
async def refresh_portfolio(request: Request, risk_profile: str = "Balanced", risk_factor: int = 5):
    """Re-fetch current market prices for the user's saved positions and
    recompute strategies from them — no re-entry/re-upload required."""
    await enforce_daily_quota(request, "portfolio/refresh")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost, name FROM portfolio_positions WHERE user_id = $1::uuid",
            user_id,
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
        return await _save_and_respond(conn, user_id, holdings_df, risk_profile, risk_factor, "Refreshed")


class PositionEditRequest(BaseModel):
    shares: float
    avg_cost: float
    name: Optional[str] = None
    risk_profile: str = "Balanced"
    risk_factor: int = 5


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
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost, name FROM portfolio_positions WHERE user_id = $1::uuid",
            user_id,
        )

        rows = []
        found = False
        for r in records:
            if r["ticker"] == ticker:
                found = True
                rows.append(
                    {"Ticker": ticker, "Name": body.name or ticker, "Shares": body.shares, "Avg_Cost": body.avg_cost}
                )
            else:
                rows.append(
                    {"Ticker": r["ticker"], "Name": r["name"] or r["ticker"], "Shares": r["shares"], "Avg_Cost": r["avg_cost"]}
                )
        if not found:
            rows.append(
                {"Ticker": ticker, "Name": body.name or ticker, "Shares": body.shares, "Avg_Cost": body.avg_cost}
            )

        holdings_df = pd.DataFrame(rows)
        return await _save_and_respond(conn, user_id, holdings_df, body.risk_profile, body.risk_factor, "Edited")


@router.get("/positions")
async def list_positions(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM portfolio_positions ORDER BY created_at DESC")
    return {"positions": [_record_to_dict(r) for r in records]}


@router.get("/strategies")
async def list_strategies(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM portfolio_strategies ORDER BY created_at DESC")
    return {"strategies": [_record_to_dict(r) for r in records]}


@router.get("/summary")
async def portfolio_summary(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM portfolio_strategies")

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


@router.get("/performance")
@limiter.limit("15/minute")
async def portfolio_performance(request: Request, lookback_days: int = 30):
    """Live portfolio value vs. what the same shares were worth `lookback_days`
    ago — always priced fresh against the market, not the last-saved snapshot."""
    await enforce_daily_quota(request, "portfolio/performance")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        records = await conn.fetch(
            "SELECT ticker, shares, avg_cost FROM portfolio_positions WHERE user_id = $1::uuid",
            user_id,
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
