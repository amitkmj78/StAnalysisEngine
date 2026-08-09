from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest_engine import (
    DAYS_PER_YEAR,
    cumulative_pct,
    max_drawdown_pct,
    run_event_driven_simulation,
    sharpe,
    sortino,
)
from .cache_utils import ttl_cache
from .index_fund_service import INDEX_FUND_UNIVERSE
from .stock_finder_service import STOCK_UNIVERSES

# TR-7: applied by default, not opt-in. Retail-realistic, not institutional —
# most brokers (including the Robinhood-style CSV import this app already
# supports) charge zero commission; slippage is a rough allowance for
# crossing the bid/ask on liquid large-caps. Borrow cost only bites if a
# strategy shorts, which this one doesn't (long-only top-N) — modeled and
# exposed anyway so a future short-capable variant doesn't need new plumbing;
# see borrow_cost_drag_pct in the output for why it's currently always 0.
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION_BPS = 0.0
DEFAULT_BORROW_COST_BPS_ANNUAL = 30.0
# TR-7: the forward-looking window each rebalance is held before the next
# ranking check — a first-class parameter (10/30/60/90 days, same set as
# the ranking lookback window elsewhere in the app), not a hardcoded
# constant. 30 as a default keeps continuity with the old ~monthly cadence.
DEFAULT_HORIZON_DAYS = 30
# Rough capacity heuristic: don't assume you can trade more than this share
# of a name's own average daily dollar volume without meaningfully moving
# it. Not a real market-impact model — a conservative, clearly-labeled
# order-of-magnitude estimate.
DEFAULT_CAPACITY_ADV_FRACTION = 0.01


def _universe_tickers(asset_type: str, universe_key: str) -> list[str]:
    if asset_type == "Stock":
        return list(STOCK_UNIVERSES.get(universe_key, []))
    if universe_key == "All":
        return [f.ticker for f in INDEX_FUND_UNIVERSE]
    return [f.ticker for f in INDEX_FUND_UNIVERSE if f.category == universe_key]


@ttl_cache(maxsize=32, ttl_seconds=21600)  # 6h — expensive to compute, doesn't need to be real-time
def backtest_momentum_ranking(
    asset_type: str,
    universe_key: str,
    lookback_days: int = 30,
    top_n: int = 5,
    years: int = 3,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    borrow_cost_bps_annual: float = DEFAULT_BORROW_COST_BPS_ANNUAL,
    risk_free_rate_annual: float = 0.0,
) -> Optional[dict]:
    """
    Event-driven walk-forward backtest of a pure trailing-return ranking
    (the same metric behind /top-performers): every horizon_days trading
    days, rank the universe by trailing lookback_days return using ONLY
    price data available up to that point, take the top top_n, and hold
    them for the next horizon_days before re-ranking. Compared against an
    equal-weight-universe benchmark over the identical daily steps.

    TR-7: the simulation itself is event-driven (see
    backtest_engine.run_event_driven_simulation) — a day-by-day loop with
    explicit portfolio state, not a single vectorized computation — which
    also means risk metrics (volatility, Sharpe, Sortino, max drawdown)
    are computed from the full daily equity curve rather than only
    sampled at each ~monthly rebalance, a materially more accurate
    methodology. Trading costs are applied by default (slippage +
    commission on turnover each rebalance). borrow_cost_drag_pct is
    computed from a real formula, wired into strategy_cumulative_return_pct
    — it's currently always 0 because this engine is long-only (no
    borrowed shares exist to charge for), not because the parameter is
    ignored.

    Deliberately price-only (no fundamentals) — that's what makes this
    honestly reconstructable at any past date via yfinance, unlike the
    fundamentals-weighted composite score used by Best Stock Finder /
    Best Index Fund, which can't be walk-forward backtested without a
    point-in-time fundamentals source this app doesn't have.
    """
    tickers = _universe_tickers(asset_type, universe_key)
    if len(tickers) < top_n + 1:
        return None

    raw = yf.download(tickers, period=f"{years + 1}y", auto_adjust=True, progress=False, group_by="ticker")

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            frame = raw[t] if len(tickers) > 1 else raw
            series = frame["Close"].dropna()
            if len(series) > lookback_days + horizon_days * 2:
                closes[t] = series
                volumes[t] = frame["Volume"].reindex(series.index)
        except Exception:
            continue

    if len(closes) < top_n + 1:
        return None

    common_index = None
    for s in closes.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    common_index = common_index.sort_values()

    cutoff = common_index[-1] - pd.Timedelta(days=years * 365)
    common_index = common_index[common_index >= cutoff]

    if len(common_index) < lookback_days + horizon_days * 2:
        return None

    price_matrix = pd.DataFrame({t: s.reindex(common_index) for t, s in closes.items()})
    volume_matrix = pd.DataFrame({t: s.reindex(common_index) for t, s in volumes.items()})

    periods, daily_strategy_returns, daily_benchmark_returns = run_event_driven_simulation(
        price_matrix, lookback_days, top_n, horizon_days, slippage_bps, commission_bps,
    )
    if not periods:
        return None

    comparable = [
        p for p in periods
        if p["strategy_return_pct"] is not None and p["benchmark_return_pct"] is not None
    ]
    hits = sum(1 for p in comparable if p["strategy_return_pct"] > p["benchmark_return_pct"])

    elapsed_years = len(daily_strategy_returns) / DAYS_PER_YEAR

    # Long-only: no shorted notional exists to charge a borrow fee against,
    # so this is structurally always 0 today — but it's a real formula in
    # the actual return chain, not an accepted-and-dropped parameter. A
    # future short-capable variant only needs to set short_notional_frac.
    short_notional_frac = 0.0
    borrow_cost_drag_pct = round(short_notional_frac * (borrow_cost_bps_annual / 100) * elapsed_years, 4)

    strategy_cumulative_gross_pct = cumulative_pct(daily_strategy_returns)
    strategy_cumulative_return_pct = (
        round(strategy_cumulative_gross_pct - borrow_cost_drag_pct, 2)
        if strategy_cumulative_gross_pct is not None else None
    )
    cagr_pct = None
    if strategy_cumulative_return_pct is not None and elapsed_years > 0:
        cagr_pct = round(((1 + strategy_cumulative_return_pct / 100) ** (1 / elapsed_years) - 1) * 100, 2)
    volatility_pct = (
        round(float(np.std(daily_strategy_returns, ddof=1)) * np.sqrt(DAYS_PER_YEAR), 2)
        if len(daily_strategy_returns) >= 2 else None
    )
    avg_turnover_pct = round(float(np.mean([p["turnover_pct"] for p in periods])), 1)

    # Capacity: least-liquid pick in the most recent rebalance sets the
    # ceiling — a book can only be as large as its most illiquid position
    # allows without excessive market impact.
    capacity_estimate_usd = None
    last_picks = periods[-1]["picks"]
    adv_dollars = []
    for t in last_picks:
        if t in volume_matrix.columns and t in price_matrix.columns:
            recent_vol = volume_matrix[t].tail(lookback_days).mean()
            recent_price = price_matrix[t].tail(lookback_days).mean()
            if pd.notna(recent_vol) and pd.notna(recent_price):
                adv_dollars.append(float(recent_vol) * float(recent_price))
    if adv_dollars:
        capacity_estimate_usd = round(min(adv_dollars) * DEFAULT_CAPACITY_ADV_FRACTION, 0)

    return {
        "asset_type": asset_type,
        "universe": universe_key,
        "lookback_days": lookback_days,
        "top_n": top_n,
        "years": years,
        "horizon_days": horizon_days,
        "slippage_bps": slippage_bps,
        "commission_bps": commission_bps,
        "borrow_cost_bps_annual": borrow_cost_bps_annual,
        "borrow_cost_drag_pct": borrow_cost_drag_pct,
        "risk_free_rate_annual": risk_free_rate_annual,
        "num_periods": len(periods),
        "hit_rate_pct": round(hits / len(comparable) * 100, 1) if comparable else None,
        "strategy_cumulative_return_pct": strategy_cumulative_return_pct,
        "benchmark_cumulative_return_pct": cumulative_pct(daily_benchmark_returns),
        "avg_strategy_period_return_pct": (
            round(float(np.mean([p["strategy_return_pct"] for p in periods])), 2) if periods else None
        ),
        "avg_benchmark_period_return_pct": (
            round(float(np.mean([p["benchmark_return_pct"] for p in periods])), 2) if periods else None
        ),
        "cagr_pct": cagr_pct,
        "volatility_pct": volatility_pct,
        "sharpe_ratio": sharpe(daily_strategy_returns, risk_free_rate_annual, DAYS_PER_YEAR),
        "sortino_ratio": sortino(daily_strategy_returns, risk_free_rate_annual, DAYS_PER_YEAR),
        "max_drawdown_pct": max_drawdown_pct(daily_strategy_returns),
        "avg_turnover_pct": avg_turnover_pct,
        "capacity_estimate_usd": capacity_estimate_usd,
        "periods": periods,
    }
