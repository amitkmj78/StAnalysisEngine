# services/portfolio_strategy.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from .yfinance_cache import get_cached_history

RISK_PROFILES = ("Conservative", "Balanced", "Aggressive")


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


_BASELINE_ANN_VOL_PCT = 28.0  # a broad single-stock-ish annualized vol to scale against


def _ticker_technicals(ticker: str) -> dict | None:
    """
    Real, ticker-specific context pulled from one shared 1y price history
    fetch (through the Alpaca-aware cache, so it benefits from the same
    resilience/dedup every other feature gets) -- used so the plan below
    varies by what THIS ticker's own price has actually been doing, not
    just the position's PnL% and the portfolio-wide risk profile/factor,
    which previously made almost every position's plan read identically.

    Returns None when there isn't enough history to say anything real
    (never fabricates a read) -- every caller below must keep working
    exactly as before when that happens, just without the extra color.
    """
    try:
        hist = get_cached_history(ticker, "1y", auto_adjust=True)
        closes = hist["Close"].dropna()
        if len(closes) < 20:
            return None

        daily_returns = closes.pct_change().dropna()
        recent_returns = daily_returns.tail(60) if len(daily_returns) >= 60 else daily_returns
        ann_vol_pct = (
            float(recent_returns.std() * (252 ** 0.5) * 100.0) if len(recent_returns) >= 10 else None
        )
        if ann_vol_pct is not None and not np.isfinite(ann_vol_pct):
            ann_vol_pct = None

        month = closes.tail(20)
        month_trend_pct = float((month.iloc[-1] / month.iloc[0] - 1.0) * 100.0) if len(month) >= 2 else None

        lo, hi = float(closes.min()), float(closes.max())
        range_position = (float(closes.iloc[-1]) - lo) / (hi - lo) if hi > lo else None

        return {
            "ann_vol_pct": ann_vol_pct,
            "month_trend_pct": month_trend_pct,
            "range_position": range_position,
        }
    except Exception:
        return None


def _volatility_multiplier(technicals: dict | None) -> float:
    """Scales the target/stop band width by this ticker's own recent
    realized volatility against a broad baseline, so a genuinely more
    volatile name gets a wider band and a steadier one a narrower band --
    instead of every position at a given risk profile getting the exact
    same percentage band regardless of the ticker. 1.0 (no adjustment)
    when volatility isn't available, which reproduces the prior
    risk-profile-only behavior exactly."""
    if not technicals or technicals.get("ann_vol_pct") is None:
        return 1.0
    vol = technicals["ann_vol_pct"]
    if vol <= 0:
        return 1.0
    return _clamp(vol / _BASELINE_ANN_VOL_PCT, 0.6, 1.8)


def _short_term_technical_clause(technicals: dict | None) -> str | None:
    """A real, ticker-specific sentence appended to the Stance line so two
    positions landing in the same PnL bucket (which otherwise get
    word-for-word identical Stance text -- see the bucket if/elif chain
    below) still read differently when their own recent price action
    differs. Based on the stock's own trailing ~1-month return, not its
    PnL vs. the user's cost basis. None when there isn't enough history
    to say anything real."""
    if not technicals or technicals.get("month_trend_pct") is None:
        return None
    trend = technicals["month_trend_pct"]
    if trend >= 8:
        return "Separately, the stock's own price action has been strongly positive over the past month."
    elif trend >= 2:
        return "Separately, the stock's own price action has drifted higher over the past month."
    elif trend <= -8:
        return "Separately, the stock's own price action has been sharply negative over the past month."
    elif trend <= -2:
        return "Separately, the stock's own price action has drifted lower over the past month."
    return "Separately, the stock's own price action has been range-bound over the past month."


def _long_term_technical_clause(technicals: dict | None) -> str | None:
    """Long-term analog of _short_term_technical_clause above: where the
    price sits in its own past-year range, a framing that reflects the
    stock's own history rather than the user's cost basis. None when
    there isn't enough history to say anything real."""
    if not technicals or technicals.get("range_position") is None:
        return None
    pos_in_range = technicals["range_position"]
    if pos_in_range >= 0.85:
        return "For context, the stock itself is trading near its 52-week high."
    elif pos_in_range <= 0.15:
        return "For context, the stock itself is trading near its 52-week low."
    return f"For context, the stock itself is trading around {pos_in_range * 100:.0f}% of the way up its 52-week range."


# -------------------------------------------------------------
# Strategy Logic
# -------------------------------------------------------------
def _compute_short_term_targets(pos: EnrichedPosition, technicals: dict | None = None) -> tuple[float, float]:
    """
    Upside target / protective stop prices for the short-term (1-4 week) plan.
    Split out from _compute_short_term_plan so callers (e.g. auto-populating
    a watchlist) can get the numeric targets without parsing the plan text.

    `technicals` (see _ticker_technicals) scales the band by this ticker's
    own realized volatility -- omitting it (the default) reproduces the
    prior risk-profile-only percentages exactly.
    """
    cp = pos.current_price
    rp = pos.risk_profile.lower()
    rf = pos.risk_factor

    if rp == "conservative":
        base_target, base_stop = 5.0, 3.0
    elif rp == "aggressive":
        base_target, base_stop = 12.0, 7.0
    else:
        base_target, base_stop = 8.0, 4.5

    vol_mult = _volatility_multiplier(technicals)
    target_pct = _clamp(base_target * vol_mult + (rf - 5) * 0.7, 3.0, 25.0)
    stop_pct = _clamp(base_stop * vol_mult + (rf - 5) * 0.4, 2.0, 18.0)

    return cp * (1 + target_pct / 100.0), cp * (1 - stop_pct / 100.0)


def _compute_short_term_plan(pos: EnrichedPosition, technicals: dict | None = None) -> str:
    """
    Generate a short-term (1–4 weeks) strategy description
    based on current PnL and risk settings.
    """
    cp = pos.current_price
    ac = pos.avg_cost
    pnl = pos.pnl_pct
    rf = pos.risk_factor

    target_price, stop_price = _compute_short_term_targets(pos, technicals)
    target_pct = (target_price / cp - 1) * 100.0
    stop_pct = (1 - stop_price / cp) * 100.0

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
    elif 0 <= pnl < 5:
        stance = (
            "PnL is roughly flat to slightly positive — not enough of a move to change your plan. "
            "Focus on whether the thesis still holds and be disciplined with your stop."
        )
    elif -10 <= pnl < 0:
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

    technical_clause = _short_term_technical_clause(technicals)
    if technical_clause:
        stance = f"{stance} {technical_clause}"

    return (
        f"**Short-Term Plan ({pos.risk_profile}, Risk {rf}/10)**\n\n"
        f"- Current price: `${cp:.2f}` (vs avg cost `${ac:.2f}` | PnL: {pnl:+.2f}%)\n"
        f"- Short-term horizon: **1–4 weeks**\n"
        f"- Upside target: **+{target_pct:.1f}%** → target price ≈ `${target_price:.2f}`\n"
        f"- Protective stop: **-{stop_pct:.1f}% from current price** → stop ≈ `${stop_price:.2f}`\n\n"
        f"**Stance:** {stance}"
    )


def _compute_long_term_plan(pos: EnrichedPosition, technicals: dict | None = None) -> str:
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
    elif 0 <= pnl < 5:
        guidance = (
            "Returns are flat to modestly positive. Long-term, the key is whether the company still "
            "fits your portfolio story (sector exposure, growth vs value, diversification). "
            "If yes, treating this as a normal fluctuation is reasonable."
        )
    elif -15 <= pnl < 0:
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

    technical_clause = _long_term_technical_clause(technicals)
    if technical_clause:
        guidance = f"{guidance} {technical_clause}"

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
                "Target_Price",
                "Stop_Price",
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

            # Fetched once per position and reused for both plans and the
            # numeric targets below, so this ticker-specific context costs
            # one shared, cached history fetch -- not three.
            technicals = _ticker_technicals(pos.ticker)
            short_plan = _compute_short_term_plan(pos, technicals)
            long_plan = _compute_long_term_plan(pos, technicals)
            target_price, stop_price = _compute_short_term_targets(pos, technicals)

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
                    "Target_Price": target_price,
                    "Stop_Price": stop_price,
                    # Passed straight through from the input row (not
                    # part of what _normalize_holdings_row/pos extract)
                    # so a preserved position's real acquired date, or a
                    # CSV import's real earliest-buy date, survives this
                    # rebuild instead of being silently dropped.
                    "Acquired_At": row.get("Acquired_At"),
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
