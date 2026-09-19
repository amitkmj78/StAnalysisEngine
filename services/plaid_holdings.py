"""
Pure mapping from Plaid's /investments/holdings/get response shape to the
Ticker/Shares/Avg_Cost DataFrame services/portfolio_strategy.py's
build_robinhood_strategies() already expects -- the same contract
services/positions_from_csv.py's compute_positions_from_trades() produces
for a CSV import, so this plugs into the exact same downstream save path
(see web/backend/plaid_sync.py).

No DB/network I/O, no plaid SDK import -- takes the already-fetched dict
services/plaid_client.fetch_holdings() returns, so this stays importable
and unit-testable with hand-built fixture dicts.
"""

from __future__ import annotations

import pandas as pd


def holdings_to_positions(holdings_response: dict) -> pd.DataFrame:
    """
    Aggregates every holding across every investment account under one
    Plaid item into one row per ticker (multi-account granularity is out
    of scope for this MVP -- a future phase could carry account_id
    through instead of summing across accounts).

    - Ticker: securities[].ticker_symbol. A holding whose security has no
      ticker_symbol (a cash sweep vehicle, an unmapped/proprietary fund)
      is dropped, not surfaced as a garbage row with a blank ticker.
    - Shares: holdings[].quantity, summed across every holding row that
      maps to the same ticker. A non-positive quantity is dropped --
      long-only, matching compute_positions_from_trades' Net_Shares > 0
      convention.
    - Avg_Cost: cost_basis / quantity per holding, weighted by quantity
      when aggregating multiple holdings into one ticker; falls back to
      institution_price for any holding where cost_basis is None (Plaid
      doesn't always have it, e.g. a transferred-in position) -- same
      "no discoverable cost -> use the current price, 0% unrealized P&L"
      convention services/manual_positions.py already uses.

    Returns an empty DataFrame (columns present, zero rows) when there's
    nothing usable, never None -- callers can hand this straight to
    build_robinhood_strategies() either way.
    """
    securities_by_id = {s["security_id"]: s for s in holdings_response.get("securities", [])}

    rows: list[dict] = []
    for holding in holdings_response.get("holdings", []):
        quantity = holding.get("quantity")
        if quantity is None or quantity <= 0:
            continue

        security = securities_by_id.get(holding.get("security_id"))
        ticker = security.get("ticker_symbol") if security else None
        if not ticker:
            continue

        cost_basis = holding.get("cost_basis")
        institution_price = holding.get("institution_price")
        per_share_cost = (
            cost_basis / quantity if cost_basis is not None and quantity else institution_price
        )
        if per_share_cost is None:
            continue

        rows.append({"Ticker": str(ticker).strip().upper(), "Shares": float(quantity), "Cost": float(per_share_cost)})

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Shares", "Avg_Cost"])

    df = pd.DataFrame(rows)
    df["TotalCost"] = df["Shares"] * df["Cost"]
    grouped = df.groupby("Ticker", as_index=False).agg(Shares=("Shares", "sum"), TotalCost=("TotalCost", "sum"))
    grouped["Avg_Cost"] = grouped["TotalCost"] / grouped["Shares"]
    return grouped[["Ticker", "Shares", "Avg_Cost"]]
