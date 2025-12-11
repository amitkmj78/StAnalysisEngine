# services/positions_from_csv.py

from __future__ import annotations

import io
from typing import IO, Dict, Any

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _to_float(series: pd.Series) -> pd.Series:
    """
    Convert strings like '$514.60', '(1,008.75)', '1.5' etc to float.
    Parentheses are treated as negative.
    """
    s = (
        series.astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.strip()
    )

    # Handle parentheses as negative
    neg_mask = s.str.startswith("(") & s.str.endswith(")")
    s = s.str.replace("(", "", regex=False).str.replace(")", "", regex=False)

    out = pd.to_numeric(s, errors="coerce")
    out[neg_mask] = -out[neg_mask]
    return out


# -----------------------------
# CSV LOADER / CLEANUP
# -----------------------------
def load_broker_activity_csv(file_obj: IO[bytes]) -> pd.DataFrame:
    """
    Read a broker / Robinhood-style 'Activity' CSV like the one you pasted and
    return a cleaned trades DataFrame with columns:

    - Ticker
    - Side  ('BUY' or 'SELL')
    - Quantity (float)
    - Price (float)

    This version is robust against:
    - Extra commas
    - Multi-line descriptions
    - Bad rows → they are skipped instead of crashing
    """
    raw = file_obj.read()

    last_err: Exception | None = None

    # Try UTF-8 then Latin-1, always with python engine & skip bad lines
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                engine="python",          # handles multiline + weird quoting
                dtype=str,
                quotechar='"',
                skipinitialspace=True,
                on_bad_lines="skip",      # <- critical: skip malformed rows
            )
            break
        except TypeError as e:
            # Older pandas: on_bad_lines not supported → fallback
            last_err = e
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    engine="python",
                    dtype=str,
                    quotechar='"',
                    skipinitialspace=True,
                    error_bad_lines=False,   # deprecated but works on older versions
                    warn_bad_lines=False,
                )
                break
            except Exception as e2:
                last_err = e2
        except Exception as e:
            last_err = e
    else:
        raise ValueError(f"Could not read CSV with any encoding/engine: {last_err}")

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Required columns (match your pasted Robinhood activity)
    required_cols = ["instrument", "trans_code", "quantity", "price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Filter to rows that are actual BUY/SELL equity trades
    df_trades = df[
        df["instrument"].notna()
        & df["instrument"].str.strip().ne("")
        & df["trans_code"].str.upper().isin(["BUY", "SELL"])
    ].copy()

    if df_trades.empty:
        raise ValueError("No BUY/SELL rows with a valid Instrument found in CSV.")

    # Clean numeric columns
    df_trades["quantity"] = _to_float(df_trades["quantity"])
    df_trades["price"] = _to_float(df_trades["price"])

    # Drop rows with invalid numbers
    df_trades = df_trades.replace([np.inf, -np.inf], np.nan)
    df_trades = df_trades.dropna(subset=["quantity", "price"])

    # Standardized output
    df_clean = pd.DataFrame(
        {
            "Ticker": df_trades["instrument"].astype(str).str.upper().str.strip(),
            "Side": df_trades["trans_code"].astype(str).str.upper().str.strip(),
            "Quantity": df_trades["quantity"].astype(float),
            "Price": df_trades["price"].astype(float),
        }
    )

    # Filter out zero-quantity noise
    df_clean = df_clean[df_clean["Quantity"] != 0].reset_index(drop=True)

    if df_clean.empty:
        raise ValueError("No valid trades left after cleaning CSV.")

    return df_clean


# -----------------------------
# POSITION RECONSTRUCTION
# -----------------------------
def compute_positions_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Given trades with columns [Ticker, Side, Quantity, Price],
    compute current open positions using an average-cost model.

    For each ticker we track:
    - Net_Shares (float)
    - Avg_Cost   (average of buys, adjusted when you add more)
    - Total_Buy_Shares
    - Total_Sell_Shares
    - Realized_PnL (approx, using average cost)
    """
    required_cols = {"Ticker", "Side", "Quantity", "Price"}
    if not required_cols.issubset(set(trades.columns)):
        raise ValueError(f"trades DataFrame must contain {required_cols}")

    positions: Dict[str, Dict[str, Any]] = {}

    for _, row in trades.iterrows():
        tk = str(row["Ticker"]).upper().strip()
        side = str(row["Side"]).upper().strip()
        qty = float(row["Quantity"])
        px = float(row["Price"])

        if tk not in positions:
            positions[tk] = {
                "Net_Shares": 0.0,
                "Avg_Cost": 0.0,
                "Total_Buy_Shares": 0.0,
                "Total_Sell_Shares": 0.0,
                "Realized_PnL": 0.0,
            }

        pos = positions[tk]

        if side == "BUY":
            old_qty = pos["Net_Shares"]
            old_cost = pos["Avg_Cost"]

            new_qty = old_qty + qty

            if new_qty <= 0:
                # Fully offset (or short) → reset avg cost
                pos["Net_Shares"] = new_qty
                pos["Avg_Cost"] = 0.0
            else:
                total_cost_before = old_qty * old_cost
                total_cost_after = total_cost_before + qty * px
                pos["Net_Shares"] = new_qty
                pos["Avg_Cost"] = total_cost_after / new_qty

            pos["Total_Buy_Shares"] += qty

        elif side == "SELL":
            sell_qty = qty
            # Realized PnL using current avg cost
            realized = (px - pos["Avg_Cost"]) * sell_qty
            pos["Realized_PnL"] += realized
            pos["Net_Shares"] -= sell_qty
            pos["Total_Sell_Shares"] += sell_qty

        # Ignore other transaction types (ACH, GDBP, GOLD, INT, etc.)

    # Turn into DataFrame
    rows = []
    for tk, p in positions.items():
        rows.append(
            {
                "Ticker": tk,
                "Net_Shares": round(p["Net_Shares"], 4),
                "Avg_Cost": round(p["Avg_Cost"], 4) if p["Net_Shares"] > 0 else np.nan,
                "Total_Buy_Shares": round(p["Total_Buy_Shares"], 4),
                "Total_Sell_Shares": round(p["Total_Sell_Shares"], 4),
                "Realized_PnL": round(p["Realized_PnL"], 2),
            }
        )

    df_pos = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)

    # You usually care only about long positions > 0
    df_open = df_pos[df_pos["Net_Shares"] > 0].reset_index(drop=True)
    return df_open


# -----------------------------
# Convenience: one-shot helper
# -----------------------------
def positions_from_activity_csv(file_obj: IO[bytes]) -> pd.DataFrame:
    """
    High-level helper:
    1) Load & clean CSV
    2) Compute open positions
    """
    trades = load_broker_activity_csv(file_obj)
    return compute_positions_from_trades(trades)
