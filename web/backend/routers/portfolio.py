import io
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.manual_positions import build_manual_positions
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

    return {
        "positions": position_rows,
        "strategies": strategy_rows,
        "summary": summarize_portfolio(strat_df),
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
