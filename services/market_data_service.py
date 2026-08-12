"""
Raw data fetch for the Market Direction Score's Internals pillar. All
yfinance I/O lives here — services/market_internals_service.py stays pure
and only consumes the DataFrames this module produces.

NOT WIRED INTO THE LIVE APP — see market_internals_service.py's module
docstring: the resulting score failed its own release-gate backtest
(contrarian, not confirming). Kept as tested, unused fetch infrastructure
in case the signal is reworked later.

Breadth (% of S&P 500 above its 50/200-day moving average) is the
expensive part: it requires several years of daily closes for every S&P
500 constituent, not just one ticker. Reuses the same S&P 500 constituent
list and parallel-fetch pattern already built for the stock screener
(services/stock_finder_service.py) rather than duplicating either.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

from services.cache_utils import ttl_cache
from services.stock_finder_service import SP500_UNIVERSE_NAME, _universe_tickers

logger = logging.getLogger(__name__)

MAX_PARALLEL_FETCHES = 10

# 11 GICS sector ETFs — used for sector relative strength (DR-I5). Not yet
# consumed by compute_internals_score (P1 doesn't build the sector
# heatmap), fetched here so it's ready when that's built.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

INTERNALS_AUX_TICKERS = ["^VIX", "^VIX3M", "XLY", "XLP", "HYG", "IEF", "RSP", "SPY"]


def _fetch_close_series(ticker: str, period: str) -> pd.Series | None:
    try:
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist.empty:
            return None
        close = hist["Close"]
        close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
        return close
    except Exception as e:
        logger.warning("Market internals: failed to fetch %s: %s", ticker, e)
        return None


def _fetch_closes_parallel(tickers: list[str], period: str) -> dict[str, pd.Series]:
    closes: dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = {executor.submit(_fetch_close_series, t, period): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            series = future.result()
            if series is not None:
                closes[ticker] = series
    return closes


@ttl_cache(maxsize=2, ttl_seconds=21600)
def fetch_sp500_breadth_history(period: str = "3y") -> pd.DataFrame:
    """
    % of S&P 500 constituents trading above their own 50-day and 200-day
    moving average, for every trading day in `period`. Cached 6h (not the
    spec's ideal 15-min internals cadence — refetching ~500 tickers' full
    history that often isn't practical on yfinance without a dedicated
    data warehouse; a known, accepted P1 limitation).
    """
    tickers = _universe_tickers(SP500_UNIVERSE_NAME)
    closes = _fetch_closes_parallel(tickers, period)
    if not closes:
        return pd.DataFrame(columns=["breadth_50dma", "breadth_200dma"])

    wide = pd.DataFrame(closes)

    def _breadth_pct(window: int) -> pd.Series:
        rolling_mean = wide.rolling(window, min_periods=window).mean()
        valid = rolling_mean.notna()  # a ticker with <window days of history yet doesn't count either way
        above = wide.gt(rolling_mean) & valid
        denom = valid.sum(axis=1).astype(float).replace(0, float("nan"))
        return above.sum(axis=1).astype(float) / denom * 100

    breadth_50 = _breadth_pct(50)
    breadth_200 = _breadth_pct(200)
    return pd.DataFrame({"breadth_50dma": breadth_50, "breadth_200dma": breadth_200})


@ttl_cache(maxsize=2, ttl_seconds=900)
def fetch_market_internals_history(period: str = "3y") -> pd.DataFrame:
    """
    Full input table for market_internals_service.compute_internals_score,
    plus a raw SPY close column (for the forward-return backtest / regime
    badge's "vs SPY" framing) — breadth, VIX level/term structure, and the
    three risk-appetite ratios, aligned on trading date.
    """
    breadth = fetch_sp500_breadth_history(period)
    aux = _fetch_closes_parallel(INTERNALS_AUX_TICKERS, period)
    if not aux or breadth.empty:
        return pd.DataFrame()

    aux_df = pd.DataFrame(aux)
    df = breadth.join(aux_df, how="inner")
    df["vix"] = df["^VIX"]
    df["vix3m"] = df["^VIX3M"]
    df["xly_xlp"] = df["XLY"] / df["XLP"]
    df["hyg_ief"] = df["HYG"] / df["IEF"]
    df["rsp_spy"] = df["RSP"] / df["SPY"]
    df["spy_close"] = df["SPY"]

    return df[
        ["breadth_50dma", "breadth_200dma", "vix", "vix3m", "xly_xlp", "hyg_ief", "rsp_spy", "spy_close"]
    ].dropna()


@ttl_cache(maxsize=2, ttl_seconds=900)
def fetch_sector_relative_strength(period: str = "1mo") -> dict[str, float]:
    """21-day return of each sector ETF minus SPY's own 21-day return —
    positive means that sector is outperforming the broad market. Not yet
    wired into the composite score; available for the sector table."""
    closes = _fetch_closes_parallel(list(SECTOR_ETFS.keys()) + ["SPY"], period)
    if "SPY" not in closes or len(closes["SPY"]) < 22:
        return {}
    spy_return = float(closes["SPY"].iloc[-1] / closes["SPY"].iloc[-22] - 1.0) * 100
    result = {}
    for ticker in SECTOR_ETFS:
        series = closes.get(ticker)
        if series is None or len(series) < 22:
            continue
        sector_return = float(series.iloc[-1] / series.iloc[-22] - 1.0) * 100
        result[ticker] = round(sector_return - spy_return, 3)
    return result
