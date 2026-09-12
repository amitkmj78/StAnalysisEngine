import threading
import time
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
from .index_fund_service import INDEX_FUND_UNIVERSE
from .rate_limit_utils import fetch_with_backoff
from .stock_finder_service import _universe_tickers as _resolve_stock_universe_tickers

# A single yf.download() call for the full "All"/S&P 500 universe
# (500+ tickers) gets hammered by Yahoo's rate limiter -- observed live:
# most tickers failing outright, and the handful that "succeeded" still
# only sharing a few months of overlapping dates instead of the
# requested multi-year window (individual rows silently rate-limited
# within an ostensibly successful response). Batching into smaller,
# paced chunks is slower but actually returns complete data instead of
# a result that LOOKS successful but silently covers a fraction of the
# requested history -- exactly the kind of thing a "3-Year Test" must
# not silently get wrong.
_DOWNLOAD_BATCH_SIZE = 50
_DOWNLOAD_BATCH_PAUSE_SECONDS = 2.0

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
        # "All" and "US - S&P 500" are deliberately empty placeholders in
        # STOCK_UNIVERSES itself (see that dict's own comment) -- they
        # resolve lazily via a live, 24h-cached Wikipedia fetch
        # (stock_finder_service.fetch_sp500_tickers), not a static lookup.
        # This function used to do `STOCK_UNIVERSES.get(universe_key, [])`
        # directly, which silently returned an empty list for both of
        # those keys -- a real, 100%-reproducible bug (not transient/rate
        # -limit related) that made every "Stock" backtest against "All"
        # or "US - S&P 500" fail with "not enough historical data",
        # regardless of horizon_days/years/lookback_days. Reusing
        # stock_finder_service's own resolver instead of reimplementing
        # it here is what keeps this from silently diverging again.
        return list(_resolve_stock_universe_tickers(universe_key))
    if universe_key == "All":
        return [f.ticker for f in INDEX_FUND_UNIVERSE]
    return [f.ticker for f in INDEX_FUND_UNIVERSE if f.category == universe_key]


def _download_universe_history(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """
    Chunked, paced replacement for one giant yf.download(tickers, ...) —
    see the module-level comment on _DOWNLOAD_BATCH_SIZE for why. Each
    chunk still goes through fetch_with_backoff for its own retry-on-
    rate-limit; a chunk that fails even after that is skipped (those
    tickers just won't appear in the result), not fatal to the whole
    universe.
    """
    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), _DOWNLOAD_BATCH_SIZE):
        chunk = tickers[i : i + _DOWNLOAD_BATCH_SIZE]
        try:
            raw = fetch_with_backoff(
                lambda c=chunk: yf.download(c, period=period, auto_adjust=True, progress=False, group_by="ticker")
            )
        except Exception:
            continue
        for t in chunk:
            try:
                frames[t] = raw[t] if len(chunk) > 1 else raw
            except Exception:
                continue
        if i + _DOWNLOAD_BATCH_SIZE < len(tickers):
            time.sleep(_DOWNLOAD_BATCH_PAUSE_SECONDS)
    return frames


_backtest_cache: dict[tuple, dict] = {}
_backtest_cache_ts: dict[tuple, float] = {}
_backtest_cache_lock = threading.Lock()
_BACKTEST_CACHE_TTL_SECONDS = 21600  # 6h — expensive to compute, doesn't need to be real-time


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
    Hand-rolled cache instead of the shared @ttl_cache decorator, on
    purpose: this must never cache a None (not-enough-data) result for
    the same 6h TTL as a real one. A transient yf.download hiccup for
    one ticker in a large universe (rate limit, timeout — exactly the
    kind of blip this app has hit repeatedly elsewhere) would otherwise
    get remembered as "impossible to backtest with these settings" for
    six hours, even though a retry moments later would likely succeed.
    Same fix shape as data_service.get_previous_close earlier this
    session. A real result is still cached for the full 6h; only a
    miss goes uncached, so the very next request retries for real.
    """
    cache_key = (
        asset_type, universe_key, lookback_days, top_n, years, horizon_days,
        slippage_bps, commission_bps, borrow_cost_bps_annual, risk_free_rate_annual,
    )
    with _backtest_cache_lock:
        cached_at = _backtest_cache_ts.get(cache_key)
        if cached_at is not None and (time.monotonic() - cached_at) < _BACKTEST_CACHE_TTL_SECONDS:
            return _backtest_cache[cache_key]

    result = _compute_backtest_momentum_ranking(
        asset_type, universe_key, lookback_days, top_n, years, horizon_days,
        slippage_bps, commission_bps, borrow_cost_bps_annual, risk_free_rate_annual,
    )

    if result is not None:
        with _backtest_cache_lock:
            _backtest_cache[cache_key] = result
            _backtest_cache_ts[cache_key] = time.monotonic()
    return result


def _compute_backtest_momentum_ranking(
    asset_type: str,
    universe_key: str,
    lookback_days: int,
    top_n: int,
    years: int,
    horizon_days: int,
    slippage_bps: float,
    commission_bps: float,
    borrow_cost_bps_annual: float,
    risk_free_rate_annual: float,
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

    frames = _download_universe_history(tickers, period=f"{years + 1}y")

    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    for t, frame in frames.items():
        try:
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
