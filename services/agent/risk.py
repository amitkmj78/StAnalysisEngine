"""Deterministic risk layer for the trading agent (no I/O).

Every order passes through preflight() before submission. Neither the
strategy nor any AI reviewer can override these rules. Every rejection,
exit, and skip carries a plain-language reason for the journal.
"""

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from services.agent.config import CONFIG, AgentConfig

NORMAL = "normal"
REDUCE_ONLY = "reduce_only"
CIRCUIT_BREAKER = "circuit_breaker"

DEFAULT_REGIME_WHEN_UNKNOWN = "Cautious"


@dataclass(frozen=True)
class Candidate:
    ticker: str
    sector: str
    price: float
    signal: str
    sma200: Optional[float]
    avg_dollar_volume: Optional[float]
    volatility_pct: Optional[float]
    earnings_blackout: bool
    short_score: Optional[float] = None


@dataclass(frozen=True)
class Holding:
    ticker: str
    sector: str
    shares: float
    price: float
    signal: Optional[str]
    sma200: Optional[float]

    @property
    def market_value(self) -> float:
        return self.shares * self.price


@dataclass
class RiskState:
    state: str
    exposure_cap_pct: float
    buys_allowed: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class Order:
    ticker: str
    side: str  # "buy" | "sell"
    qty: int
    est_price: float
    est_value: float
    trigger: str
    reason: str


def regime_cap_pct(regime: Optional[str], config: AgentConfig = CONFIG) -> tuple[float, str]:
    """Returns (cap, reason). A missing regime reading uses the conservative
    Cautious cap and says so, rather than assuming Risk-On."""
    if regime in config.regime_exposure_pct:
        return config.regime_exposure_pct[regime], f"Regime {regime} caps exposure at {config.regime_exposure_pct[regime]:g}%."
    cap = config.regime_exposure_pct[DEFAULT_REGIME_WHEN_UNKNOWN]
    return cap, f"No regime reading; using the conservative {DEFAULT_REGIME_WHEN_UNKNOWN} cap of {cap:g}%."


def risk_state(
    equity: float,
    last_equity: Optional[float],
    peak_equity: Optional[float],
    breaker_latched: bool,
    regime_cap: float,
    config: AgentConfig = CONFIG,
) -> RiskState:
    """AGT-17 daily loss limit, AGT-18 drawdown breaker. Breaker, once
    latched, stays until a manual reset (the caller persists the latch)."""
    reasons: list[str] = []

    if breaker_latched:
        return RiskState(
            CIRCUIT_BREAKER, min(regime_cap, config.drawdown_breaker_exposure_pct), False,
            ["Drawdown circuit breaker is latched; exposure capped and buys blocked until a manual reset."],
        )

    if peak_equity and peak_equity > 0:
        drawdown_pct = (equity / peak_equity - 1.0) * 100.0
        if drawdown_pct <= -config.drawdown_breaker_pct:
            return RiskState(
                CIRCUIT_BREAKER, min(regime_cap, config.drawdown_breaker_exposure_pct), False,
                [f"Equity is {drawdown_pct:.1f}% below its peak; drawdown breaker tripped (limit -{config.drawdown_breaker_pct:g}%)."],
            )

    if last_equity and last_equity > 0:
        day_pct = (equity / last_equity - 1.0) * 100.0
        if day_pct <= -config.daily_loss_limit_pct:
            return RiskState(
                REDUCE_ONLY, regime_cap, False,
                [f"Equity is {day_pct:.2f}% vs the prior close; daily loss limit (-{config.daily_loss_limit_pct:g}%) hit, so buys are blocked for the rest of the day."],
            )

    return RiskState(NORMAL, regime_cap, True, reasons)


def filter_candidates(
    candidates: Sequence[Candidate], config: AgentConfig = CONFIG
) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """AGT-5..8: only Buy signals, above the 200-day average, liquid, and
    outside the earnings blackout. Returns (passed, [(ticker, reason)])."""
    passed: list[Candidate] = []
    rejected: list[tuple[str, str]] = []
    for c in candidates:
        if c.signal != "Buy":
            rejected.append((c.ticker, f"Signal is {c.signal}, not Buy."))
        elif c.sma200 is None:
            rejected.append((c.ticker, "Not enough price history for a 200-day average."))
        elif c.price <= c.sma200:
            rejected.append((c.ticker, f"Price {c.price:.2f} is at or below its 200-day average {c.sma200:.2f}."))
        elif c.avg_dollar_volume is None or c.avg_dollar_volume < config.min_avg_dollar_volume:
            rejected.append((c.ticker, f"20-day average dollar volume is below ${config.min_avg_dollar_volume / 1e6:.0f}M."))
        elif c.earnings_blackout:
            rejected.append((c.ticker, f"Earnings within {config.earnings_blackout_trading_days} trading days; new buys skipped."))
        elif c.volatility_pct is None or c.volatility_pct <= 0:
            rejected.append((c.ticker, "Not enough return history to size the position."))
        else:
            passed.append(c)
    return passed, rejected


def size_positions(
    candidates: Sequence[Candidate],
    equity: float,
    exposure_cap_pct: float,
    config: AgentConfig = CONFIG,
) -> list[dict]:
    """AGT-11..13: inverse-volatility weights over the top candidates, then
    the per-name and per-sector caps. Candidates must already be sorted
    best-first; only the first max_positions are sized."""
    chosen = list(candidates)[: config.max_positions]
    if not chosen or equity <= 0:
        return []

    inv_vol = {c.ticker: 1.0 / c.volatility_pct for c in chosen}
    total_inv = sum(inv_vol.values())
    invested_budget = equity * exposure_cap_pct / 100.0
    name_cap = equity * config.max_position_pct / 100.0
    sector_cap = equity * config.max_sector_pct / 100.0

    targets: dict[str, float] = {}
    for c in chosen:
        raw = invested_budget * inv_vol[c.ticker] / total_inv
        targets[c.ticker] = min(raw, name_cap)

    by_sector: dict[str, list[str]] = {}
    for c in chosen:
        by_sector.setdefault(c.sector, []).append(c.ticker)
    for sector, tickers in by_sector.items():
        total = sum(targets[t] for t in tickers)
        if total > sector_cap and total > 0:
            scale = sector_cap / total
            for t in tickers:
                targets[t] *= scale

    result = []
    for c in chosen:
        value = targets[c.ticker]
        result.append({
            "ticker": c.ticker,
            "sector": c.sector,
            "target_value": round(value, 2),
            "weight_pct": round(value / equity * 100.0, 4),
            "volatility_pct": c.volatility_pct,
            "price": c.price,
        })
    return result


def scale_for_vol_target(
    weights: Mapping[str, float], returns: pd.DataFrame, target_annual_pct: float
) -> tuple[float, Optional[float]]:
    """AGT-14: estimated annualized portfolio volatility from the covariance
    of daily returns. Returns (scale factor in (0, 1], estimate). Volatility
    scales linearly with exposure, so scaling by target/estimate is exact."""
    tickers = [t for t in weights if t in returns.columns]
    if not tickers:
        return 1.0, None
    w = np.array([weights[t] for t in tickers], dtype=float)
    cov = returns[tickers].dropna().cov().values * 252.0
    variance = float(w @ cov @ w)
    if variance <= 0 or not math.isfinite(variance):
        return 1.0, None
    estimate = math.sqrt(variance) * 100.0
    if estimate <= target_annual_pct:
        return 1.0, round(estimate, 2)
    return target_annual_pct / estimate, round(estimate, 2)


def exit_triggers(holding: Holding, targets: set[str], config: AgentConfig = CONFIG) -> list[tuple[str, str]]:
    """AGT-9: returns [(rule, plain-language reason)] for every exit rule that fires."""
    fired: list[tuple[str, str]] = []
    if holding.signal == "Trim":
        fired.append(("signal_trim", "Signal dropped to Trim."))
    if holding.sma200 is not None and holding.price < holding.sma200:
        fired.append(("below_200dma", f"Price {holding.price:.2f} closed below its 200-day average {holding.sma200:.2f}."))
    if holding.ticker not in targets:
        fired.append(("left_top_set", "No longer among the top-ranked Buy candidates."))
    return fired


def plan_orders(
    holdings: Sequence[Holding],
    sized: Sequence[dict],
    equity: float,
    state: RiskState,
    config: AgentConfig = CONFIG,
) -> tuple[list[Order], list[tuple[str, str]]]:
    """Proposed orders, sells first (AGT-20). Returns (orders, skipped
    [(ticker, reason)]). Exits are full sells with the triggering rule
    recorded; rebalances smaller than the band produce no order (AGT-10)."""
    targets = {s["ticker"]: s for s in sized}
    holding_map = {h.ticker: h for h in holdings}
    sells: list[Order] = []
    buys: list[Order] = []
    skipped: list[tuple[str, str]] = []
    band_value = equity * config.rebalance_band_pct / 100.0

    for h in holdings:
        triggers = exit_triggers(h, set(targets), config)
        if triggers:
            qty = int(math.floor(h.shares))
            if qty > 0:
                sells.append(Order(
                    h.ticker, "sell", qty, h.price, round(qty * h.price, 2),
                    triggers[0][0], "; ".join(reason for _, reason in triggers),
                ))
            continue

        t = targets.get(h.ticker)
        if t is None:
            continue
        delta_value = t["target_value"] - h.market_value
        if abs(delta_value) < band_value:
            skipped.append((h.ticker, f"Position is within {config.rebalance_band_pct:g}% of target; no trade."))
            continue
        if delta_value > 0:
            if not state.buys_allowed:
                skipped.append((h.ticker, state.reasons[0] if state.reasons else "Buys are blocked."))
                continue
            qty = int(math.floor(delta_value / h.price))
            if qty > 0:
                buys.append(Order(h.ticker, "buy", qty, h.price, round(qty * h.price, 2), "rebalance_up", "Add toward the target weight."))
        else:
            qty = int(math.floor(-delta_value / h.price))
            if qty > 0:
                sells.append(Order(h.ticker, "sell", qty, h.price, round(qty * h.price, 2), "rebalance_down", "Trim toward the target weight."))

    held = set(holding_map)
    for s in sized:
        if s["ticker"] in held:
            continue
        if not state.buys_allowed:
            skipped.append((s["ticker"], state.reasons[0] if state.reasons else "Buys are blocked."))
            continue
        qty = int(math.floor(s["target_value"] / s["price"]))
        if qty <= 0:
            skipped.append((s["ticker"], "Target weight is smaller than one share."))
            continue
        buys.append(Order(
            s["ticker"], "buy", qty, s["price"], round(qty * s["price"], 2), "new_entry",
            f"New Buy signal passed all filters; entering at its inverse-volatility weight "
            f"({s['weight_pct']:g}% of equity, sized from {s['volatility_pct']:.1f}% annualized volatility).",
        ))

    return sells + buys, skipped


def preflight(
    order: Order,
    *,
    holdings: Sequence[Holding],
    cash: float,
    equity: float,
    state: RiskState,
    exposure_cap_pct: float,
    sector_by_ticker: Mapping[str, str],
    config: AgentConfig = CONFIG,
) -> Optional[str]:
    """AGT-19. Returns None if the order passes, else the specific reason
    it must be rejected. Called sequentially, so `cash` and `holdings`
    should reflect fills admitted earlier in the same run."""
    held = {h.ticker: h for h in holdings}
    current = held.get(order.ticker)
    held_shares = current.shares if current else 0.0

    if order.side == "sell":
        if order.qty > held_shares + 1e-9:
            return f"Sell of {order.qty} exceeds the {held_shares:g} shares held (no shorting)."
        return None

    if order.side != "buy":
        return f"Unknown order side '{order.side}'."
    if not state.buys_allowed:
        return state.reasons[0] if state.reasons else "Buys are blocked by the current risk state."
    if order.est_value > cash + 1e-6:
        return f"Buy of ${order.est_value:,.0f} exceeds available cash plus same-run sale proceeds (${cash:,.0f})."

    new_position_value = (current.market_value if current else 0.0) + order.est_value
    if new_position_value > equity * config.max_position_pct / 100.0 + 1e-6:
        return f"Position would reach ${new_position_value:,.0f}, above the {config.max_position_pct:g}% single-stock cap."

    sector = sector_by_ticker.get(order.ticker, "Unknown")
    sector_value = sum(h.market_value for h in holdings if sector_by_ticker.get(h.ticker, "Unknown") == sector)
    if sector_value + order.est_value > equity * config.max_sector_pct / 100.0 + 1e-6:
        return f"{sector} would exceed the {config.max_sector_pct:g}% sector cap."

    gross = sum(h.market_value for h in holdings)
    if gross + order.est_value > equity * exposure_cap_pct / 100.0 + 1e-6:
        return f"Total exposure would exceed the {exposure_cap_pct:g}% cap for the current regime and risk state."

    if current is None and len(held) >= config.max_positions:
        return f"Already at the {config.max_positions}-position cap."

    return None
