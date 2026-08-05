from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .cache_utils import ttl_cache
from .index_fund_service import INDEX_FUND_UNIVERSE
from .stock_finder_service import STOCK_UNIVERSES

REBALANCE_TRADING_DAYS = 21  # ~monthly
PERIODS_PER_YEAR = 252 / REBALANCE_TRADING_DAYS  # ~12, for annualizing period stats

# TR-7: applied by default, not opt-in. Retail-realistic, not institutional —
# most brokers (including the Robinhood-style CSV import this app already
# supports) charge zero commission; slippage is a rough allowance for
# crossing the bid/ask on liquid large-caps. Borrow cost only bites if a
# strategy shorts, which this one doesn't (long-only top-N) — modeled and
# exposed anyway so a future short-capable variant doesn't need new plumbing.
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION_BPS = 0.0
DEFAULT_BORROW_COST_BPS_ANNUAL = 30.0
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


def _cumulative_pct(period_returns_pct: list[float]) -> Optional[float]:
    if not period_returns_pct:
        return None
    total = 1.0
    for r in period_returns_pct:
        total *= 1 + r / 100
    return round((total - 1) * 100, 2)


def _max_drawdown_pct(period_returns_pct: list[float]) -> Optional[float]:
    if not period_returns_pct:
        return None
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in period_returns_pct:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)
    return round(max_dd * 100, 2)


def _sharpe(period_returns_pct: list[float], risk_free_rate_annual: float) -> Optional[float]:
    if len(period_returns_pct) < 2:
        return None
    returns = np.array(period_returns_pct) / 100
    rf_per_period = risk_free_rate_annual / PERIODS_PER_YEAR
    excess = returns - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std == 0:
        return None
    return round(float(np.mean(excess)) / std * np.sqrt(PERIODS_PER_YEAR), 2)


def _sortino(period_returns_pct: list[float], risk_free_rate_annual: float) -> Optional[float]:
    if len(period_returns_pct) < 2:
        return None
    returns = np.array(period_returns_pct) / 100
    rf_per_period = risk_free_rate_annual / PERIODS_PER_YEAR
    excess = returns - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None  # never had a losing period — Sortino is undefined, not infinite
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(np.abs(downside[0]))
    if downside_std == 0:
        return None
    return round(float(np.mean(excess)) / downside_std * np.sqrt(PERIODS_PER_YEAR), 2)


@ttl_cache(maxsize=32, ttl_seconds=21600)  # 6h — expensive to compute, doesn't need to be real-time
def backtest_momentum_ranking(
    asset_type: str,
    universe_key: str,
    lookback_days: int = 30,
    top_n: int = 5,
    years: int = 3,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    borrow_cost_bps_annual: float = DEFAULT_BORROW_COST_BPS_ANNUAL,
    risk_free_rate_annual: float = 0.0,
) -> Optional[dict]:
    """
    Walk-forward backtest of a pure trailing-return ranking (the same
    metric behind /top-performers): at each ~monthly rebalance point over
    the last `years` years, rank the universe by trailing `lookback_days`
    return using ONLY price data available up to that point, take the top
    `top_n`, and measure their actual forward return until the next
    rebalance. Compared against an equal-weight-universe benchmark over
    the identical periods.

    TR-7: trading costs are applied by default (slippage + commission on
    turnover each rebalance; long-only here so borrow cost is modeled but
    currently always zero-impact), and results report CAGR, annualized
    volatility, Sharpe, Sortino, max drawdown, average turnover, and a
    rough capacity estimate alongside the existing hit rate and cumulative
    return figures. `strategy_return_pct` in each period is net of costs;
    `strategy_return_gross_pct` is what it would have been cost-free, so
    the cost drag is visible rather than hidden.

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
            if len(series) > lookback_days + REBALANCE_TRADING_DAYS * 2:
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

    if len(common_index) < lookback_days + REBALANCE_TRADING_DAYS * 2:
        return None

    price_matrix = pd.DataFrame({t: s.reindex(common_index) for t, s in closes.items()})
    volume_matrix = pd.DataFrame({t: s.reindex(common_index) for t, s in volumes.items()})

    rebalance_points = list(
        range(lookback_days, len(common_index) - REBALANCE_TRADING_DAYS, REBALANCE_TRADING_DAYS)
    )
    if not rebalance_points:
        return None

    round_trip_cost_pct = 2 * (slippage_bps + commission_bps) / 100  # bps -> %, doubled for entry+exit

    periods = []
    prev_picks: Optional[set[str]] = None
    for i in rebalance_points:
        lookback_start = price_matrix.iloc[i - lookback_days]
        lookback_end = price_matrix.iloc[i]
        momentum = (lookback_end / lookback_start - 1.0).dropna()
        if len(momentum) < top_n:
            continue
        picks = momentum.sort_values(ascending=False).head(top_n).index.tolist()
        picks_set = set(picks)

        # Turnover: share of the book that changed hands this rebalance —
        # the first period is full turnover (the initial purchase).
        turnover_frac = 1.0 if prev_picks is None else len(picks_set - prev_picks) / len(picks_set)
        prev_picks = picks_set

        forward_end_idx = min(i + REBALANCE_TRADING_DAYS, len(common_index) - 1)
        forward_start = price_matrix.iloc[i]
        forward_end = price_matrix.iloc[forward_end_idx]

        pick_returns = (forward_end[picks] / forward_start[picks] - 1.0).dropna()
        strategy_return_gross = float(pick_returns.mean() * 100) if not pick_returns.empty else None
        cost_drag_pct = turnover_frac * round_trip_cost_pct
        strategy_return = (
            round(strategy_return_gross - cost_drag_pct, 2) if strategy_return_gross is not None else None
        )

        universe_returns = (forward_end / forward_start - 1.0).dropna()
        benchmark_return = float(universe_returns.mean() * 100) if not universe_returns.empty else None

        periods.append({
            "date": common_index[i].strftime("%Y-%m-%d"),
            "picks": picks,
            "strategy_return_pct": strategy_return,
            "strategy_return_gross_pct": round(strategy_return_gross, 2) if strategy_return_gross is not None else None,
            "benchmark_return_pct": round(benchmark_return, 2) if benchmark_return is not None else None,
            "turnover_pct": round(turnover_frac * 100, 1),
        })

    if not periods:
        return None

    strategy_returns = [p["strategy_return_pct"] for p in periods if p["strategy_return_pct"] is not None]
    benchmark_returns = [p["benchmark_return_pct"] for p in periods if p["benchmark_return_pct"] is not None]
    comparable = [
        p for p in periods
        if p["strategy_return_pct"] is not None and p["benchmark_return_pct"] is not None
    ]
    hits = sum(1 for p in comparable if p["strategy_return_pct"] > p["benchmark_return_pct"])

    strategy_cumulative_return_pct = _cumulative_pct(strategy_returns)
    cagr_pct = None
    if strategy_cumulative_return_pct is not None and years > 0:
        cagr_pct = round(((1 + strategy_cumulative_return_pct / 100) ** (1 / years) - 1) * 100, 2)
    volatility_pct = (
        round(float(np.std(strategy_returns, ddof=1)) * np.sqrt(PERIODS_PER_YEAR), 2)
        if len(strategy_returns) >= 2 else None
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
        "slippage_bps": slippage_bps,
        "commission_bps": commission_bps,
        "borrow_cost_bps_annual": borrow_cost_bps_annual,
        "risk_free_rate_annual": risk_free_rate_annual,
        "num_periods": len(periods),
        "hit_rate_pct": round(hits / len(comparable) * 100, 1) if comparable else None,
        "strategy_cumulative_return_pct": strategy_cumulative_return_pct,
        "benchmark_cumulative_return_pct": _cumulative_pct(benchmark_returns),
        "avg_strategy_period_return_pct": round(float(np.mean(strategy_returns)), 2) if strategy_returns else None,
        "avg_benchmark_period_return_pct": round(float(np.mean(benchmark_returns)), 2) if benchmark_returns else None,
        "cagr_pct": cagr_pct,
        "volatility_pct": volatility_pct,
        "sharpe_ratio": _sharpe(strategy_returns, risk_free_rate_annual),
        "sortino_ratio": _sortino(strategy_returns, risk_free_rate_annual),
        "max_drawdown_pct": _max_drawdown_pct(strategy_returns),
        "avg_turnover_pct": avg_turnover_pct,
        "capacity_estimate_usd": capacity_estimate_usd,
        "periods": periods,
    }
