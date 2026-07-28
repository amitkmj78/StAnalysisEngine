import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from services.trade_storage import _evaluate_single_trade
from services.trade_strategy_service import compute_trade_strategy

from web.backend.auth import verify_bearer_token
from web.backend.db import user_conn
from web.backend.rate_limit import enforce_daily_quota, limiter

router = APIRouter(prefix="/api/v1/trades", tags=["trades"], dependencies=[Depends(verify_bearer_token)])

# services/trade_storage.py::_evaluate_single_trade expects the original
# SQLite schema's CapitalCase column names (row["Entry_Low"], etc.) and
# returns a dict with the same casing. The Postgres table uses lowercase
# snake_case (Postgres convention). Translate both directions rather than
# duplicate the (tested) evaluation logic under a different key scheme.
_TO_LEGACY = {
    "trade_id": "Trade_ID", "ticker": "Ticker", "direction": "Direction",
    "strategy_type": "Strategy_Type", "created_at": "Created_At",
    "entry_low": "Entry_Low", "entry_high": "Entry_High", "stop_loss": "Stop_Loss",
    "target": "Target", "context": "Context", "risk_profile": "Risk_Profile",
    "risk_factor": "Risk_Factor", "status": "Status", "entry_price": "Entry_Price",
    "entry_date": "Entry_Date", "exit_price": "Exit_Price", "exit_date": "Exit_Date",
    "max_runup_pct": "Max_Runup_Pct", "max_drawdown_pct": "Max_Drawdown_Pct",
    "realized_pnl_pct": "Realized_PnL_Pct", "days_in_trade": "Days_In_Trade",
}
_FROM_LEGACY = {v: k for k, v in _TO_LEGACY.items()}

_ROW_COLUMNS = [
    "trade_id", "ticker", "direction", "strategy_type", "created_at",
    "entry_low", "entry_high", "stop_loss", "target", "context", "risk_profile",
    "risk_factor", "status", "entry_price", "entry_date", "exit_price", "exit_date",
    "max_runup_pct", "max_drawdown_pct", "realized_pnl_pct", "days_in_trade",
]


def _record_to_dict(record) -> dict:
    return {k: record[k] for k in record.keys()}


def _row_to_legacy_series(row: dict) -> pd.Series:
    mapped = {_TO_LEGACY.get(k, k): v for k, v in row.items()}
    return pd.Series(mapped)


def _legacy_result_to_row(evaluated: dict) -> dict:
    out = {}
    for key, value in evaluated.items():
        pg_key = _FROM_LEGACY.get(key)
        if pg_key is None:
            continue
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        elif isinstance(value, np.floating):
            value = float(value)
        elif isinstance(value, np.integer):
            value = int(value)
        elif isinstance(value, float) and np.isnan(value):
            value = None
        out[pg_key] = value
    return out


class TradeCreate(BaseModel):
    ticker: str
    entry_low: float
    entry_high: float
    stop_loss: float
    target: float
    direction: str = "LONG"
    strategy_type: str = "Discretionary"
    context: str = ""
    risk_profile: str = ""
    risk_factor: Optional[float] = None


@router.post("")
@limiter.limit("20/minute")
async def create_trade(request: Request, body: TradeCreate):
    await enforce_daily_quota(request, "trades/create")
    user_id = request.state.user["id"]

    created_at = datetime.now(timezone.utc)
    trade_id = f"{body.ticker.upper()}_{int(created_at.timestamp())}_{uuid.uuid4().hex[:6]}"

    async with user_conn(user_id) as conn:
        record = await conn.fetchrow(
            """
            INSERT INTO trades (
                trade_id, user_id, ticker, direction, strategy_type, created_at,
                entry_low, entry_high, stop_loss, target, context, risk_profile,
                risk_factor, status
            ) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'OPEN')
            RETURNING *
            """,
            trade_id, user_id, body.ticker.upper(), body.direction.upper(), body.strategy_type,
            created_at, body.entry_low, body.entry_high, body.stop_loss, body.target,
            body.context, body.risk_profile, body.risk_factor,
        )
    return {"trade": _record_to_dict(record)}


@router.get("")
async def list_trades(request: Request):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        records = await conn.fetch("SELECT * FROM trades ORDER BY created_at DESC")
    trades = [_record_to_dict(r) for r in records]

    # get_latest_price is ttl_cache'd (services/data_service.py), so repeat
    # list loads within the cache window don't re-hit yfinance per ticker —
    # dedupe here too so one page load never fetches the same ticker twice.
    unique_tickers = {t["ticker"] for t in trades}
    prices = {
        ticker: await run_in_threadpool(get_latest_price, ticker)
        for ticker in unique_tickers
    }
    for trade in trades:
        trade["current_price"] = prices.get(trade["ticker"])
        trade.update(compute_trade_strategy(trade))

    return {"trades": trades}


@router.post("/evaluate")
@limiter.limit("10/minute")
async def evaluate_trades_endpoint(request: Request):
    await enforce_daily_quota(request, "trades/evaluate")
    user_id = request.state.user["id"]

    async with user_conn(user_id) as conn:
        existing = await conn.fetch("SELECT * FROM trades")
        if not existing:
            return {"trades": []}

        as_of = datetime.now(timezone.utc)
        updated_rows = []
        for record in existing:
            row = _record_to_dict(record)
            legacy_series = _row_to_legacy_series(row)
            evaluated = await run_in_threadpool(_evaluate_single_trade, legacy_series, as_of)
            updated_rows.append(_legacy_result_to_row(evaluated))

        # Per-row update (not a destructive whole-table replace like the
        # original evaluate_trades()) — RLS + explicit filter means this only
        # ever touches this user's own rows.
        update_columns = [c for c in _ROW_COLUMNS if c != "trade_id"]
        for row in updated_rows:
            # Built from the same filtered list in one pass — SET placeholder
            # numbers must line up positionally with `values`, or asyncpg
            # can't infer parameter types for placeholders the query text
            # never actually references (IndeterminateDatatypeError).
            set_clause = ", ".join(f"{col} = ${i + 2}" for i, col in enumerate(update_columns))
            values = [row.get(col) for col in update_columns]
            await conn.execute(
                f"UPDATE trades SET {set_clause} WHERE trade_id = $1",
                row["trade_id"], *values,
            )

    return {"trades": updated_rows}


@router.delete("/{trade_id}")
async def delete_trade(request: Request, trade_id: str):
    user_id = request.state.user["id"]
    async with user_conn(user_id) as conn:
        await conn.execute("DELETE FROM trades WHERE trade_id = $1", trade_id)
    return {"deleted": trade_id}
