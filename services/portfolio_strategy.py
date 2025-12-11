# services/portfolio_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


# -------------------------------------------------------------
# Data structure for an enriched position
# -------------------------------------------------------------
@dataclass
class EnrichedPosition:
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    pnl_pct: float
    risk_profile: str
    risk_factor: int


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _get_live_price(ticker: str) -> float:
    """
    Basic live/last close fetch using yfinance.
    If anything fails, returns NaN.
    """
    try:
        hist = yf.Ticker(ticker).history(period="1d", auto_adjust=True)
        if hist.empty:
            return float("nan")
        return float(hist["Close"].iloc[-1])
    except Exception:
        return float("nan")


# -------------------------------------------------------------
# Strategy Logic
# -------------------------------------------------------------
def _compute_short_term_plan(pos: EnrichedPosition) -> str:
    """
    Generate a short-term (1–4 weeks) strategy description
    based on current PnL and risk settings.
    """
    cp = pos.current_price
    ac = pos.avg_cost
    pnl = pos.pnl_pct
    rp = pos.risk_profile.lower()
    rf = pos.risk_factor

    # ---- Base target/stop by risk profile ----
    if rp == "conservative":
        base_target = 5.0
        base_stop = 3.0
    elif rp == "aggressive":
        base_target = 12.0
        base_stop = 7.0
    else:  # balanced / default
        base_target = 8.0
        base_stop = 4.5

    # Fine-tune using risk_factor (1–10)
    # Higher risk_factor → larger target, looser stop
    target_pct = base_target + (rf - 5) * 0.7
    stop_pct = base_stop + (rf - 5) * 0.4

    target_pct = _clamp(target_pct, 3.0, 20.0)
    stop_pct = _clamp(stop_pct, 2.0, 15.0)

    # Short-term target and stop are from CURRENT price (not avg cost)
    target_price = cp * (1 + target_pct / 100.0)
    stop_price = cp * (1 - stop_pct / 100.0)

    # Behaviour depending on PnL
    if pnl >= 20:
        stance = (
            "You're sitting on strong gains. Consider taking partial profits "
            "into strength while letting a core position run with a tighter trailing stop."
        )
    elif 5 <= pnl < 20:
        stance = (
            "Position is working. You can keep holding, but think about defining a level "
            "where you'd trim if momentum fades."
        )
    elif -10 <= pnl < 5:
        stance = (
            "PnL is roughly flat to slightly negative. Focus on whether the thesis "
            "still holds and be disciplined with your stop."
        )
    elif -25 <= pnl < -10:
        stance = (
            "This is a meaningful drawdown. Avoid adding purely to 'average down' "
            "unless your conviction and time horizon are very strong."
        )
    else:  # pnl < -25
        stance = (
            "Deep drawdown territory. You should re-evaluate the thesis honestly and "
            "decide whether to cut risk, reduce size, or exit."
        )

    return (
        f"**Short-Term Plan ({pos.risk_profile}, Risk {rf}/10)**\n\n"
        f"- Current price: `${cp:.2f}` (vs avg cost `${ac:.2f}` | PnL: {pnl:+.2f}%)\n"
        f"- Short-term horizon: **1–4 weeks**\n"
        f"- Upside target: **+{target_pct:.1f}%** → target price ≈ `${target_price:.2f}`\n"
        f"- Protective stop: **-{stop_pct:.1f}% from current price** → stop ≈ `${stop_price:.2f}`\n\n"
        f"**Stance:** {stance}"
    )


def _compute_long_term_plan(pos: EnrichedPosition) -> str:
    """
    Generate a long-term (6–24 months) strategy description.
    """
    cp = pos.current_price
    ac = pos.avg_cost
    pnl = pos.pnl_pct
    rp = pos.risk_profile.lower()
    rf = pos.risk_factor

    if rp == "conservative":
        horizon = "12–24 months"
        trim_trigger = 20.0
    elif rp == "aggressive":
        horizon = "6–18 months"
        trim_trigger = 35.0
    else:  # balanced
        horizon = "9–24 months"
        trim_trigger = 25.0

    if pnl >= trim_trigger:
        guidance = (
            "The position has significantly out-performed your cost basis. "
            "Define a long-term thesis (earnings growth, moat, macro tailwind) and consider "
            "trimming a portion to lock in gains while keeping exposure as long as fundamentals stay intact."
        )
    elif pnl >= 5:
        guidance = (
            "You're ahead on the position. As long as the business story is intact, "
            "you can keep holding and use fundamentals (revenue/earnings trends, margins, "
            "competitive position) as your primary decision anchors."
        )
    elif -15 <= pnl < 5:
        guidance = (
            "Returns are flat to modestly negative. Long-term, the key is whether the company still "
            "fits your portfolio story (sector exposure, growth vs value, diversification). "
            "If yes, treating this as a normal fluctuation is reasonable."
        )
    elif -35 <= pnl < -15:
        guidance = (
            "This is a sizable long-term drawdown. Before averaging down, revisit the fundamentals: "
            "has something structurally changed (earnings, debt, competition, regulation)? "
            "If the thesis is weakened, it can be better to reduce or exit rather than hope."
        )
    else:  # pnl < -35
        guidance = (
            "The position is deeply underwater from your cost basis. Long-term recovery is only realistic "
            "if the business is still fundamentally sound. Otherwise, crystallizing the loss and reallocating "
            "to stronger names can be the rational choice."
        )

    risk_note = {
        "conservative": (
            "Because you're conservative, concentrate on durable businesses, strong balance sheets, and "
            "avoid oversized single-stock bets."
        ),
        "aggressive": (
            "With an aggressive profile, it's fine to accept volatility, but size positions such that a single "
            "blow-up doesn't wreck the overall portfolio."
        ),
    }.get(rp, "Balance growth names with a core of stable holdings to smooth overall volatility.")

    return (
        f"**Long-Term Plan ({pos.risk_profile}, Risk {rf}/10)**\n\n"
        f"- Investment horizon: **{horizon}**\n"
        f"- Current vs cost: `${cp:.2f}` vs `${ac:.2f}` (PnL: {pnl:+.2f}%)\n\n"
        f"{guidance}\n\n"
        f"**Risk framing:** {risk_note}"
    )


# -------------------------------------------------------------
# PUBLIC API: build_robinhood_strategies + summarize_portfolio
# -------------------------------------------------------------
def _normalize_holdings_row(row: pd.Series, risk_profile: str, risk_factor: int) -> EnrichedPosition:
    """
    Take a row from holdings_df (from CSV or manual input)
    and normalize to EnrichedPosition.
    Expected possible columns in holdings_df:
      - 'Ticker'
      - 'Shares' or 'Net_Shares'
      - 'Avg_Cost'
      - 'Current_Price' (optional — we’ll fetch if missing)
      - 'Unrealized_PnL_%' (optional — we’ll compute if missing)
    """
    ticker = str(row.get("Ticker", "")).upper().strip()
    if not ticker:
        raise ValueError("Ticker missing in holdings_df row")

    # Shares
    if "Shares" in row.index:
        shares = _safe_float(row["Shares"])
    else:
        shares = _safe_float(row.get("Net_Shares", 0.0))

    # Avg cost
    avg_cost = _safe_float(row.get("Avg_Cost", 0.0))

    # Current price
    cur_price = _safe_float(row.get("Current_Price", float("nan")))
    if not np.isfinite(cur_price) or cur_price <= 0:
        cur_price = _get_live_price(ticker)

    if not np.isfinite(cur_price) or cur_price <= 0:
        # Fallback: use avg_cost if all else fails
        cur_price = avg_cost

    # PnL %
    if "Unrealized_PnL_%" in row.index and pd.notna(row["Unrealized_PnL_%"]):
        pnl_pct = _safe_float(row["Unrealized_PnL_%"])
    else:
        if avg_cost > 0:
            pnl_pct = (cur_price - avg_cost) / avg_cost * 100.0
        else:
            pnl_pct = 0.0

    return EnrichedPosition(
        ticker=ticker,
        shares=shares,
        avg_cost=avg_cost,
        current_price=cur_price,
        pnl_pct=pnl_pct,
        risk_profile=risk_profile,
        risk_factor=risk_factor,
    )


def build_robinhood_strategies(
    holdings_df: pd.DataFrame,
    risk_profile: str = "Balanced",
    risk_factor: int = 5,
) -> pd.DataFrame:
    """
    Main entry point used by the app.

    Input: holdings_df with at least:
      - 'Ticker'
      - 'Shares' or 'Net_Shares'
      - 'Avg_Cost'

    Output: DataFrame with:
      - Ticker
      - Shares
      - Avg_Cost
      - Current_Price
      - Unrealized_PnL_%
      - Short_Term_Plan
      - Long_Term_Plan
    """
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "Avg_Cost",
                "Current_Price",
                "Unrealized_PnL_%",
                "Short_Term_Plan",
                "Long_Term_Plan",
            ]
        )

    rp = risk_profile.capitalize()
    rf = int(_clamp(risk_factor, 1, 10))

    rows = []
    for _, row in holdings_df.iterrows():
        try:
            pos = _normalize_holdings_row(row, rp, rf)
            if pos.shares <= 0:
                continue

            short_plan = _compute_short_term_plan(pos)
            long_plan = _compute_long_term_plan(pos)

            rows.append(
                {
                    "Ticker": pos.ticker,
                    "Shares": pos.shares,
                    "Avg_Cost": pos.avg_cost,
                    "Current_Price": pos.current_price,
                    "Unrealized_PnL_%" : pos.pnl_pct,
                    "Short_Term_Plan": short_plan,
                    "Long_Term_Plan": long_plan,
                    "Risk_Profile": rp,
                    "Risk_Factor": rf,
                }
            )
        except Exception:
            # Skip broken rows; optionally add logging
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Shares",
                "Avg_Cost",
                "Current_Price",
                "Unrealized_PnL_%",
                "Short_Term_Plan",
                "Long_Term_Plan",
                "Risk_Profile",
                "Risk_Factor",
            ]
        )

    df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
    return df


def summarize_portfolio(strat_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic summary stats from the strategy DataFrame.
    """
    if strat_df is None or strat_df.empty:
        return {
            "total_positions": 0,
            "total_value": 0.0,
            "total_pnl_pct": 0.0,
        }

    total_positions = len(strat_df)

    # Portfolio market value = sum(Shares * Current_Price)
    total_value = float((strat_df["Shares"] * strat_df["Current_Price"]).sum())

    # Approx portfolio-level PnL% (value-weighted)
    weights = (strat_df["Shares"] * strat_df["Current_Price"])
    if weights.sum() > 0:
        total_pnl_pct = float((weights * strat_df["Unrealized_PnL_%"] / 100.0).sum() / weights.sum() * 100.0)
    else:
        total_pnl_pct = 0.0

    return {
        "total_positions": total_positions,
        "total_value": total_value,
        "total_pnl_pct": total_pnl_pct,
    }
