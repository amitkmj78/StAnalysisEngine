from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .cache_utils import ttl_cache
from .index_fund_service import INDEX_FUND_UNIVERSE
from .stock_finder_service import STOCK_UNIVERSES

REBALANCE_TRADING_DAYS = 21  # ~monthly


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


@ttl_cache(maxsize=32, ttl_seconds=21600)  # 6h — expensive to compute, doesn't need to be real-time
def backtest_momentum_ranking(
    asset_type: str,
    universe_key: str,
    lookback_days: int = 30,
    top_n: int = 5,
    years: int = 3,
) -> Optional[dict]:
    """
    Walk-forward backtest of a pure trailing-return ranking (the same
    metric behind /top-performers): at each ~monthly rebalance point over
    the last `years` years, rank the universe by trailing `lookback_days`
    return using ONLY price data available up to that point, take the top
    `top_n`, and measure their actual forward return until the next
    rebalance. Compared against an equal-weight-universe benchmark over
    the identical periods.

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
    for t in tickers:
        try:
            series = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
            if len(series) > lookback_days + REBALANCE_TRADING_DAYS * 2:
                closes[t] = series
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

    rebalance_points = list(
        range(lookback_days, len(common_index) - REBALANCE_TRADING_DAYS, REBALANCE_TRADING_DAYS)
    )
    if not rebalance_points:
        return None

    periods = []
    for i in rebalance_points:
        lookback_start = price_matrix.iloc[i - lookback_days]
        lookback_end = price_matrix.iloc[i]
        momentum = (lookback_end / lookback_start - 1.0).dropna()
        if len(momentum) < top_n:
            continue
        picks = momentum.sort_values(ascending=False).head(top_n).index.tolist()

        forward_end_idx = min(i + REBALANCE_TRADING_DAYS, len(common_index) - 1)
        forward_start = price_matrix.iloc[i]
        forward_end = price_matrix.iloc[forward_end_idx]

        pick_returns = (forward_end[picks] / forward_start[picks] - 1.0).dropna()
        strategy_return = float(pick_returns.mean() * 100) if not pick_returns.empty else None

        universe_returns = (forward_end / forward_start - 1.0).dropna()
        benchmark_return = float(universe_returns.mean() * 100) if not universe_returns.empty else None

        periods.append({
            "date": common_index[i].strftime("%Y-%m-%d"),
            "picks": picks,
            "strategy_return_pct": round(strategy_return, 2) if strategy_return is not None else None,
            "benchmark_return_pct": round(benchmark_return, 2) if benchmark_return is not None else None,
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

    return {
        "asset_type": asset_type,
        "universe": universe_key,
        "lookback_days": lookback_days,
        "top_n": top_n,
        "years": years,
        "num_periods": len(periods),
        "hit_rate_pct": round(hits / len(comparable) * 100, 1) if comparable else None,
        "strategy_cumulative_return_pct": _cumulative_pct(strategy_returns),
        "benchmark_cumulative_return_pct": _cumulative_pct(benchmark_returns),
        "avg_strategy_period_return_pct": round(float(np.mean(strategy_returns)), 2) if strategy_returns else None,
        "avg_benchmark_period_return_pct": round(float(np.mean(benchmark_returns)), 2) if benchmark_returns else None,
        "periods": periods,
    }
