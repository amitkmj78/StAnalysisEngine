# services/portfolio_storage.py

from __future__ import annotations
import os
import sqlite3
import pandas as pd
from datetime import datetime

# ------------------------------------------
# DB LOCATION
# ------------------------------------------
# Global DB path used everywhere
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "trades.db")
DB_PATH = os.path.abspath(DB_PATH)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(BASE_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(BASE_DIR, "portfolio.db")


# ------------------------------------------
# Helper: Auto-migrate SQLite tables
# ------------------------------------------
def delete_all_rows(table_name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table_name}")
    conn.commit()
    conn.close()

def delete_row_by_id(table_name: str, id_column: str, row_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table_name} WHERE {id_column} = ?", (row_id,))
    conn.commit()
    conn.close()

def load_table(table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def _ensure_table_columns(table: str, required_cols: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # 1. Ensure table exists
    cols_sql = ", ".join([f"{col} {dtype}" for col, dtype in required_cols.items()])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols_sql}
        );
    """)

    # 2. Check existing columns
    cur.execute(f"PRAGMA table_info({table});")
    existing_cols = [row[1] for row in cur.fetchall()]

    # 3. Add missing columns
    for col, dtype in required_cols.items():
        if col not in existing_cols:
            print(f"[MIGRATION] Adding column '{col}' to '{table}'")
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {dtype}")

    con.commit()
    con.close()


def init_portfolio_tables():
    """Ensure both tables exist and are schema-synchronized."""
    _ensure_table_columns("portfolio_positions", {
        "Ticker": "TEXT",
        "Name": "TEXT",
        "Shares": "REAL",
        "Avg_Cost": "REAL",
        "Current_Price": "REAL",
        "Total_Return": "REAL",
        "Total_Buy_Shares": "REAL",
        "Total_Sell_Shares": "REAL",
        "Source": "TEXT",
        "Created_At": "TEXT"
    })

    _ensure_table_columns("portfolio_strategies", {
        "Ticker": "TEXT",
        "Shares": "REAL",
        "Avg_Cost": "REAL",
        "Current_Price": "REAL",
        "Unrealized_PnL": "REAL",
        "Short_Term_Plan": "TEXT",
        "Long_Term_Plan": "TEXT",
        "Risk_Profile": "TEXT",
        "Risk_Factor": "INTEGER",
        "Created_At": "TEXT"
    })


# Run migration at import
init_portfolio_tables()


# ------------------------------------------
# SAVE PORTFOLIO POSITIONS
# ------------------------------------------
def save_portfolio_positions(holdings_df: pd.DataFrame, source: str):
    """Save Robinhood / Manual positions into the SQLite table."""
    #fix_portfolio_strategies_schema()
    fix_portfolio_positions_schema()
    con = sqlite3.connect(DB_PATH)

    # ensure consistent columns
    holdings_df = holdings_df.copy()
    holdings_df["Source"] = source
    holdings_df["Created_At"] = datetime.now()

    holdings_df.to_sql(
        "portfolio_positions",
        con,
        if_exists="append",
        index=False
    )

    con.close()


# ------------------------------------------
# LOAD PORTFOLIO POSITIONS
# ------------------------------------------
def load_portfolio_positions() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolio_positions ORDER BY ID DESC", con)
    con.close()
    return df


# ------------------------------------------
# SAVE STRATEGIES (Robinhood + Manual)
# ------------------------------------------
def save_portfolio_strategies(strat_df: pd.DataFrame, risk_profile: str, risk_factor: int):
    fix_portfolio_strategies_schema()
    con = sqlite3.connect(DB_PATH)

    df = strat_df.copy()
    df["Risk_Profile"] = risk_profile
    df["Risk_Factor"] = risk_factor
    df["Created_At"] = datetime.utcnow().isoformat()

    df.to_sql(
        "portfolio_strategies",
        con,
        if_exists="append",
        index=False
    )

    con.close()


# ------------------------------------------
# LOAD STRATEGIES
# ------------------------------------------
def load_portfolio_strategies() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM portfolio_strategies ORDER BY ID DESC", con)
    con.close()
    return df

def fix_portfolio_strategies_schema():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Fetch existing columns
    cur.execute("PRAGMA table_info(portfolio_strategies)")
    existing = [row[1] for row in cur.fetchall()]

    # Required columns for strategies table
    required = {
        "Strategy_ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "Ticker": "TEXT",
        "Strategy_Name": "TEXT",
        "Risk_Profile": "TEXT",
        "Risk_Factor": "REAL DEFAULT 1",
        "Entry": "REAL",
        "Stop": "REAL",
        "Target": "REAL",
        "Realized_PnL": "REAL DEFAULT 0",
        "Unrealized_PnL": "REAL DEFAULT 0",
        "`Unrealized_PnL_%`": "REAL DEFAULT 0",  # <-- NOTE BACKTICKS!
        "`Realized_PnL_%`": "REAL DEFAULT 0"     # <-- prevent SQL errors
    }

    # Try to add missing columns
    for col, ddl in required.items():
        clean_col = col.replace("`", "")  # remove backticks for logical comparison

        if clean_col not in existing:
            try:
                cur.execute(f"ALTER TABLE portfolio_strategies ADD COLUMN {col} {ddl}")
                print(f"[DB] Added missing column: {clean_col}")
            except Exception as e:
                print(f"[DB] Could NOT add {clean_col}: {e}")

    conn.commit()
    conn.close()

def fix_portfolio_positions_schema():
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Get existing columns
        cur.execute("PRAGMA table_info(portfolio_positions)")
        existing = [row[1] for row in cur.fetchall()]

        required = {
            "Position_ID": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "Ticker": "TEXT",
            "Quantity": "REAL",
            "Avg_Cost": "REAL",
            "Realized_PnL": "REAL DEFAULT 0",
            "Unrealized_PnL": "REAL DEFAULT 0",
            "`Realized_PnL_%`": "REAL DEFAULT 0",
            "`Unrealized_PnL_%`": "REAL DEFAULT 0",
            "Source": "TEXT"
        }

        for col, ddl in required.items():
            clean_col = col.replace("`", "")

            if clean_col not in existing:
                try:
                    cur.execute(f"ALTER TABLE portfolio_positions ADD COLUMN {col} {ddl}")
                    print(f"[DB] Added missing column: {clean_col}")
                except Exception as e:
                    print(f"[DB] Could NOT add column {clean_col}: {e}")

        conn.commit()
        conn.close()


