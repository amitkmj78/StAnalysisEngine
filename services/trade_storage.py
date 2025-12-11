# services/trade_storage.py

from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
#   DATA DIRECTORY + DB
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "trades.db")
CSV_PATH = os.path.join(DATA_DIR, "trade_ideas_log.csv")

def _ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
#   SCHEMA (ONLY PLACE WHERE COLUMNS EXIST)
# ============================================================
TRADE_SCHEMA = {
    "Trade_ID": "TEXT PRIMARY KEY",
    "Ticker": "TEXT",
    "Direction": "TEXT",
    "Strategy_Type": "TEXT",
    "Created_At": "TEXT",
    "Entry_Low": "REAL",
    "Entry_High": "REAL",
    "Stop_Loss": "REAL",
    "Target": "REAL",
    "Context": "TEXT",
    "Risk_Profile": "TEXT",
    "Risk_Factor": "REAL",
    "Status": "TEXT",
    "Entry_Price": "REAL",
    "Entry_Date": "TEXT",
    "Exit_Price": "REAL",
    "Exit_Date": "TEXT",
    "Max_Runup_Pct": "REAL",
    "Max_Drawdown_Pct": "REAL",
    "Realized_PnL_Pct": "REAL",
    "Days_In_Trade": "REAL",
}

# Auto-generate SQL
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS trades (
    {", ".join([f"{col} {ctype}" for col, ctype in TRADE_SCHEMA.items()])}
);
"""


# ============================================================
#   INIT DATABASE
# ============================================================
def _init_db():
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    conn.close()

_init_db()


# ============================================================
#   LOAD TRADES
# ============================================================
def load_trades() -> pd.DataFrame:
    _init_db()
    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql_query("SELECT * FROM trades", conn)
    except Exception:
        df = pd.DataFrame(columns=list(TRADE_SCHEMA.keys()))

    conn.close()

    # Parse datetime fields
    for c in ["Created_At", "Entry_Date", "Exit_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


# ============================================================
#   CHECK IF TRADE EXISTS (Avoid duplicates)
# ============================================================
def _trade_exists(trade_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM trades WHERE Trade_ID=?", (trade_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


# ============================================================
#   SAVE TRADE (DYNAMIC SQL)
# ============================================================
def save_trade_idea(
    ticker: str,
    entry_low: float,
    entry_high: float,
    stop_loss: float,
    target: float,
    direction="LONG",
    strategy_type="Discretionary",
    context: str = "",
    risk_profile: str = "",
    risk_factor: float | None = None,
    created_at: Optional[datetime] = None,
):

    _init_db()
    created_at = created_at or datetime.utcnow()
    trade_id = f"{ticker}_{int(created_at.timestamp())}"

    if _trade_exists(trade_id):
        print(f"⚠ Trade already exists — skipping {trade_id}")
        return

    # Build row dynamically from schema
    row = {col: None for col in TRADE_SCHEMA.keys()}

    row.update({
        "Trade_ID": trade_id,
        "Ticker": ticker.upper(),
        "Direction": direction.upper(),
        "Strategy_Type": strategy_type,
        "Created_At": created_at.isoformat(),
        "Entry_Low": entry_low,
        "Entry_High": entry_high,
        "Stop_Loss": stop_loss,
        "Target": target,
        "Context": context,
        "Risk_Profile": risk_profile,
        "Risk_Factor": risk_factor,
        "Status": "OPEN",
    })

    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?" for _ in row.values()])

    conn = sqlite3.connect(DB_PATH)
    conn.execute(f"INSERT INTO trades ({cols}) VALUES ({placeholders})", tuple(row.values()))
    conn.commit()
    conn.close()

    print(f"✅ Saved trade {trade_id}")


# ============================================================
#   EVALUATE A SINGLE TRADE
# ============================================================
def _evaluate_single_trade(row: pd.Series, as_of: datetime) -> Dict[str, Any]:

    ticker = row["Ticker"]
    created_at = pd.to_datetime(row["Created_At"])

    start = created_at.date()
    end = as_of.date() + timedelta(days=1)

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    except:
        return row.to_dict()

    if hist.empty:
        return row.to_dict()

    entry_low = row["Entry_Low"]
    entry_high = row["Entry_High"]
    stop = row["Stop_Loss"]
    tgt = row["Target"]

    # Find entry
    entry_date = None
    entry_price = None

    for dt, high, low in zip(hist.index, hist["High"], hist["Low"]):
        if low <= entry_high and high >= entry_low:
            entry_date = dt
            entry_price = (entry_low + entry_high) / 2
            break

    if entry_date is None:
        return {**row.to_dict(), "Status": "OPEN"}

    # After entry
    after = hist[hist.index >= entry_date]

    exit_date = None
    exit_price = None
    status = "ACTIVE"

    for dt, high, low in zip(after.index, after["High"], after["Low"]):

        if low <= stop:
            exit_date = dt
            exit_price = stop
            status = "STOP_HIT"
            break

        if high >= tgt:
            exit_date = dt
            exit_price = tgt
            status = "TARGET_HIT"
            break

    # Expiration logic
    if exit_date is None:
        if (after.index[-1] - entry_date).days > 60:
            exit_date = after.index[-1]
            exit_price = after["Close"].iloc[-1]
            status = "EXPIRED"
        else:
            return {
                **row.to_dict(),
                "Status": "ACTIVE",
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
            }

    pnl = (exit_price - entry_price) / entry_price * 100.0

    return {
        **row.to_dict(),
        "Status": status,
        "Entry_Date": entry_date,
        "Entry_Price": entry_price,
        "Exit_Date": exit_date,
        "Exit_Price": exit_price,
        "Realized_PnL_Pct": round(pnl, 2),
    }


# ============================================================
#   EVALUATE AND SAVE BACK
# ============================================================
def evaluate_trades(as_of: Optional[datetime] = None) -> pd.DataFrame:
    as_of = as_of or datetime.utcnow()
    df = load_trades()

    if df.empty:
        return df

    updated_rows = [_evaluate_single_trade(row, as_of) for _, row in df.iterrows()]
    df_new = pd.DataFrame(updated_rows)

    conn = sqlite3.connect(DB_PATH)
    df_new.to_sql("trades", conn, if_exists="replace", index=False)
    conn.close()

    return df_new

