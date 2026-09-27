from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import ta

from services.backtest_engine import max_drawdown_pct
from services.cache_utils import ttl_cache
from services.screener_service import INDEX_MAP
from services.yfinance_cache import get_cached_history, get_cached_info

logger = logging.getLogger(__name__)

SP500_UNIVERSE_NAME = "US - S&P 500"
# 10 concurrent unpaced yfinance requests (2 calls each: history + info)
# was a real, observed trigger for Yahoo's rate limiter — a burst that size
# fires effectively instantly since ThreadPoolExecutor has no pacing
# between workers. Lower concurrency plus fetch_with_backoff's per-call
# pacing (see _build_stock_row) trades some wall-clock time on a full
# S&P 500 scan for not tripping a sustained, account-wide block.
MAX_PARALLEL_FETCHES = 4

# "All" and SP500_UNIVERSE_NAME resolve their ticker lists lazily via
# _universe_tickers (a live, cached Wikipedia fetch) rather than at import
# time — a blocked/slow network call must never delay app startup. These
# keys exist here as empty placeholders purely so /universes listing and
# the router's `universe in STOCK_UNIVERSES` validation keep working.
STOCK_UNIVERSES: Dict[str, List[str]] = {
    "All": [],
    SP500_UNIVERSE_NAME: [],
    **INDEX_MAP,
}

# Reused as a safety net if the live S&P 500 fetch fails (network issue,
# Wikipedia page structure change) — degrade to a smaller known-good list
# rather than error out or silently return zero results.
_SP500_FALLBACK = INDEX_MAP["US - Mega Cap (SPY sample)"]


@ttl_cache(maxsize=4, ttl_seconds=86400)
def fetch_sp500_tickers() -> List[str]:
    """
    Live S&P 500 constituent list from Wikipedia, cached 24h. This app has
    no other source of real index membership — every other universe here is
    a small hardcoded sample. Membership drifts (~20-30 changes/year); this
    fetch keeps it current without needing a manually-maintained list.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; StAnalysisEngine/1.0)"},
            timeout=10,
        )
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text))[0]
        # yfinance uses a hyphen for share classes (BRK-B), Wikipedia a dot (BRK.B).
        symbols = table["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        tickers = sorted({s for s in symbols if s})
        return tickers if len(tickers) > 400 else _SP500_FALLBACK
    except Exception as e:
        logger.warning("Could not fetch S&P 500 constituent list, using fallback: %s", e)
        return _SP500_FALLBACK


def _universe_tickers(universe_key: str) -> List[str]:
    if universe_key == SP500_UNIVERSE_NAME:
        return fetch_sp500_tickers()
    if universe_key == "All":
        return sorted({t for group in INDEX_MAP.values() for t in group} | set(fetch_sp500_tickers()))
    return STOCK_UNIVERSES.get(universe_key, [])


# Yahoo's own sector taxonomy is an 11-sector partition, same count as
# GICS but with different names for 6 of them -- a rename, not new data.
# The other 5 (Communication Services, Industrials, Energy, Real Estate,
# Utilities) are already identical strings in both systems.
GICS_SECTOR_RENAME: Dict[str, str] = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Healthcare": "Health Care",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}

GICS_SECTORS_ORDER = [
    "Information Technology", "Health Care", "Financials", "Consumer Discretionary",
    "Communication Services", "Industrials", "Consumer Staples", "Energy",
    "Utilities", "Real Estate", "Materials",
]


def _gics_sector(raw_sector: str | None) -> str:
    if not raw_sector:
        return "Unknown"
    return GICS_SECTOR_RENAME.get(raw_sector, raw_sector)


LOWER_IS_BETTER = {"volatility_6m", "max_drawdown_1y", "max_drawdown_3y", "forward_pe"}

METRIC_LABELS: Dict[str, str] = {
    "return_1m": "1-Month Return",
    "return_3m": "3-Month Return",
    "return_6m": "6-Month Return",
    "return_1y": "1-Year Return",
    "return_3y_annualized": "3-Year Annualized Return",
    "rsi_balance": "RSI Balance",
    "macd_signal_strength": "MACD Signal Strength",
    "volume_strength": "Volume Strength",
    "volatility_6m": "6-Month Volatility",
    "max_drawdown_1y": "1-Year Max Drawdown",
    "max_drawdown_3y": "3-Year Max Drawdown",
    "sharpe_3y": "3-Year Sharpe Ratio",
    "forward_pe": "Forward P/E",
    "revenue_growth": "Revenue Growth",
    "earnings_growth": "Earnings Growth",
}

METRIC_UNITS: Dict[str, str] = {
    "return_1m": "%",
    "return_3m": "%",
    "return_6m": "%",
    "return_1y": "%",
    "return_3y_annualized": "%",
    "rsi_balance": "pts",
    "macd_signal_strength": "pts",
    "volume_strength": "%",
    "volatility_6m": "%",
    "max_drawdown_1y": "%",
    "max_drawdown_3y": "%",
    "sharpe_3y": "",
    "forward_pe": "x",
    "revenue_growth": "%",
    "earnings_growth": "%",
}


GOAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Short Term": {
        "return_1m": 0.25,
        "return_3m": 0.30,
        "rsi_balance": 0.15,
        "macd_signal_strength": 0.15,
        "volume_strength": 0.10,
        "volatility_6m": 0.05,
    },
    # Reweighted away from recent-momentum windows (return_6m dropped
    # entirely, return_1y cut to a light residual) toward what "long term"
    # is actually supposed to mean: the full-history return and risk-
    # adjusted return (sharpe_3y), full-history drawdown rather than just
    # the trailing year, and the existing valuation/quality metrics --
    # previously 60% of this score was some flavor of recent price
    # momentum, which is what "Short Term" already measures; MRNA/AMD
    # showing up in both lists was a direct symptom of that overlap.
    "Long Term": {
        "return_3y_annualized": 0.25,
        "sharpe_3y": 0.25,
        "max_drawdown_3y": 0.15,
        "revenue_growth": 0.10,
        "earnings_growth": 0.10,
        "forward_pe": 0.10,
        "return_1y": 0.05,
    },
}


def _pct_return(close: pd.Series, lookback: int) -> float | None:
    if close.empty or len(close) <= lookback:
        return None
    start = float(close.iloc[-lookback - 1])
    end = float(close.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100


def _annualized_return(close: pd.Series, trading_days: int = 252, min_years: float = 2.9) -> float | None:
    """
    Compound annual growth rate over `close`'s full span. Requires at
    least ~min_years of real trading history before annualizing at all --
    without this, a short window gets raised to a large power (1/years),
    which is exactly how a recently-spun-off/IPO'd ticker with only a few
    months of real history can show a "3-year annualized return" in the
    thousands of percent. Same "not enough data yet -> None" convention
    _pct_return already uses below, not a new pattern.

    Also sanity-bounded: an annualized figure this extreme is far more
    likely a computation artifact (or a non-repeatable one-off event) than
    a reliable signal worth ranking #1 on, so it's flagged out (None)
    rather than silently trusted and fed into scoring.
    """
    if close.empty or len(close) < 2:
        return None
    years = len(close) / trading_days
    if years < min_years:
        return None
    total_return = float(close.iloc[-1]) / float(close.iloc[0])
    annualized = (total_return ** (1 / years) - 1.0) * 100
    if annualized > 200 or annualized < -95:
        return None
    return annualized


def _max_drawdown(close: pd.Series) -> float | None:
    if close.empty:
        return None
    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    return abs(float(drawdown.min())) * 100


def _safe_percent(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value * 100 if abs(value) <= 1 else value


def _score_series(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([0.0] * len(series), index=series.index)

    min_val = numeric.min()
    max_val = numeric.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        scaled = pd.Series([1.0] * len(series), index=series.index)
    else:
        scaled = (numeric - min_val) / (max_val - min_val)

    if lower_is_better:
        scaled = 1 - scaled

    mean_val = scaled.mean()
    return scaled.fillna(mean_val if not pd.isna(mean_val) else 0.0)


def _rsi_balance_score(rsi: float | None) -> float | None:
    if rsi is None or pd.isna(rsi):
        return None
    # Prefer not-overheated but still strong momentum.
    return max(0.0, 100 - abs(float(rsi) - 55) * 3)


def _build_stock_row(ticker_symbol: str) -> dict | None:
    try:
        # Shared cache (services/yfinance_cache.py): dedupes against the
        # Fund Screener, Goal Plan, entry-strategy scanner, etc. pulling
        # the same ticker's history/info within the same 15-minute window.
        hist = get_cached_history(ticker_symbol, "3y", auto_adjust=True)
        info = get_cached_info(ticker_symbol)

        if hist.empty or len(hist) < 70:
            return None

        close = hist["Close"]
        volume = hist["Volume"] if "Volume" in hist.columns else pd.Series(dtype=float)
        latest_price = float(close.iloc[-1])

        return_1m = _pct_return(close, 21)
        return_3m = _pct_return(close, 63)
        return_6m = _pct_return(close, 126)
        return_1y = _pct_return(close, 252)
        return_3y_annualized = _annualized_return(close)

        # Literal trading-day trailing windows for the Top Performers
        # leaderboard — deliberately separate from the 1M/3M/6M/1Y columns
        # above (21/63/126/252-day approximations used by the composite
        # scoring system) so the two features can't drift into confusing
        # near-duplicates of each other.
        return_10d = _pct_return(close, 10)
        return_30d = _pct_return(close, 30)
        return_60d = _pct_return(close, 60)
        return_90d = _pct_return(close, 90)
        max_drawdown_1y = _max_drawdown(close.tail(252))
        max_drawdown_3y = _max_drawdown(close)

        daily_returns_6m = close.tail(126).pct_change().dropna()
        volatility_6m = (
            float(daily_returns_6m.std() * np.sqrt(252) * 100)
            if not daily_returns_6m.empty
            else None
        )

        # Risk-adjusted return over the fund's/stock's full available
        # history (up to 3y), risk-free rate treated as 0% -- same
        # simplifying convention services/momentum_backtest_service.py
        # already uses, not a new assumption. None (not a fake 0) when
        # there's too little history or zero volatility to divide by.
        daily_returns_3y = close.pct_change().dropna()
        if return_3y_annualized is not None and not daily_returns_3y.empty:
            volatility_3y = float(daily_returns_3y.std() * np.sqrt(252) * 100)
            sharpe_3y = (return_3y_annualized / volatility_3y) if volatility_3y else None
        else:
            sharpe_3y = None

        try:
            rsi = float(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1])
        except Exception:
            rsi = None

        try:
            macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
            macd_value = float(macd_indicator.macd().iloc[-1])
            macd_signal = float(macd_indicator.macd_signal().iloc[-1])
            macd_signal_strength = (macd_value - macd_signal) * 100
        except Exception:
            macd_signal_strength = None

        try:
            recent_volume = float(volume.iloc[-1])
            avg_volume = float(volume.tail(20).mean())
            volume_strength = ((recent_volume / avg_volume) - 1.0) * 100 if avg_volume else None
        except Exception:
            volume_strength = None

        return {
            "Ticker": ticker_symbol,
            "Name": info.get("shortName") or info.get("longName") or ticker_symbol,
            "Sector": info.get("sector") or "Unknown",
            "GICS Sector": _gics_sector(info.get("sector")),
            "Industry": info.get("industry") or "Unknown",
            "Last Close Date": close.index[-1].date().isoformat(),
            "Price": latest_price,
            "Market Cap ($B)": (
                float(info.get("marketCap")) / 1_000_000_000
                if info.get("marketCap")
                else None
            ),
            "Forward PE": info.get("forwardPE"),
            "Revenue Growth %": _safe_percent(info.get("revenueGrowth")),
            "Earnings Growth %": _safe_percent(info.get("earningsGrowth")),
            "1M Return %": return_1m,
            "3M Return %": return_3m,
            "6M Return %": return_6m,
            "1Y Return %": return_1y,
            "3Y Annualized %": return_3y_annualized,
            "Return 10D %": return_10d,
            "Return 30D %": return_30d,
            "Return 60D %": return_60d,
            "Return 90D %": return_90d,
            "RSI": rsi,
            "RSI Balance": _rsi_balance_score(rsi),
            "MACD Strength": macd_signal_strength,
            "Volume Strength %": volume_strength,
            "6M Volatility %": volatility_6m,
            "1Y Max Drawdown %": max_drawdown_1y,
            "3Y Max Drawdown %": max_drawdown_3y,
            "3Y Sharpe": sharpe_3y,
        }
    except Exception:
        return None


@ttl_cache(maxsize=64, ttl_seconds=3600)
def get_stock_finder_table(universe_key: str) -> pd.DataFrame:
    tickers = _universe_tickers(universe_key)
    rows: List[dict] = []

    # Each _build_stock_row is a couple of independent, I/O-bound yfinance
    # calls — parallelize so a 500-ticker universe (S&P 500) is tractable.
    # Order doesn't matter here since rank_stocks sorts the result afterward.
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_FETCHES) as executor:
        futures = [executor.submit(_build_stock_row, ticker) for ticker in tickers]
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)


# Maps a compare-page/momentum window code to the already-computed
# return column in get_stock_finder_table's output -- "1Y"/365 reuses the
# existing "1Y Return %" column (there is no separate "Return 365D %"
# column, and none is needed).
WINDOW_RETURN_COLUMN = {
    "10D": "Return 10D %", "30D": "Return 30D %", "60D": "Return 60D %", "90D": "Return 90D %", "1Y": "1Y Return %",
}


def rank_stocks_by_window_return(
    window_code: str, universe_key: str, top_n: int, owned_tickers: Optional[set] = None
) -> List[dict]:
    """
    Top-N stocks by trailing return over `window_code`, shared by
    GET /momentum/top-performers (Stock asset_type) and the /portfolio/
    compare endpoint's top_stocks -- one ranking, so the two surfaces
    can't silently disagree. `owned` flags whichever tickers the caller's
    own portfolio holds; spark/signal are left for the caller to fill in
    (this function stays FastAPI/DB-free, no per-ticker signal lookups).
    """
    col = WINDOW_RETURN_COLUMN.get(window_code)
    if col is None:
        raise ValueError(f"window_code must be one of {sorted(WINDOW_RETURN_COLUMN)}")
    df = get_stock_finder_table(universe_key)
    if df.empty or col not in df.columns:
        return []
    owned_tickers = owned_tickers or set()
    ranked = df.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
    return [
        {
            "rank": i + 1,
            "ticker": row["Ticker"],
            "name": row["Name"],
            "sector": _gics_sector(row["Sector"]),
            "return_pct": round(float(row[col]), 2),
            "owned": row["Ticker"] in owned_tickers,
            "spark": [],
            "signal": None,
        }
        for i, row in ranked.iterrows()
    ]


@ttl_cache(maxsize=64, ttl_seconds=3600)
def get_single_stock_table(ticker_symbol: str) -> pd.DataFrame:
    cleaned = ticker_symbol.strip().upper()
    if not cleaned:
        return pd.DataFrame()
    row = _build_stock_row(cleaned)
    return pd.DataFrame([row]) if row is not None else pd.DataFrame()


def rank_stocks(goal: str, universe_key: str) -> pd.DataFrame:
    df = get_stock_finder_table(universe_key).copy()
    if df.empty:
        return df

    df["return_1m"] = df["1M Return %"]
    df["return_3m"] = df["3M Return %"]
    df["return_6m"] = df["6M Return %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["rsi_balance"] = df["RSI Balance"]
    df["macd_signal_strength"] = df["MACD Strength"]
    df["volume_strength"] = df["Volume Strength %"]
    df["volatility_6m"] = df["6M Volatility %"]
    df["max_drawdown_1y"] = df["1Y Max Drawdown %"]
    df["max_drawdown_3y"] = df["3Y Max Drawdown %"]
    df["sharpe_3y"] = df["3Y Sharpe"]
    df["forward_pe"] = df["Forward PE"]
    df["revenue_growth"] = df["Revenue Growth %"]
    df["earnings_growth"] = df["Earnings Growth %"]

    weights = GOAL_WEIGHTS[goal]
    total_score = pd.Series([0.0] * len(df), index=df.index)

    for metric, weight in weights.items():
        total_score += _score_series(df[metric], lower_is_better=metric in LOWER_IS_BETTER) * weight

    df["Score"] = (total_score * 100).round(1)

    secondary_sort = "3M Return %" if goal == "Short Term" else "1Y Return %"
    # Ticker-alpha as the final tie-break makes basket selection fully
    # deterministic (DI-02: "ties broken by market cap, then ticker") --
    # Score is rounded to 1 decimal, so exact ties are common enough to
    # matter for a top-N-per-sector cut.
    return (
        df.sort_values(
            ["Score", secondary_sort, "Market Cap ($B)", "Ticker"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def get_basket_candidates(goal: str, universe_key: str) -> tuple[pd.DataFrame, List[dict], str]:
    """
    Classifies every ticker in universe_key into eligible-for-the-basket
    or excluded-with-a-reason (DI-03) -- unlike rank_stocks/
    get_stock_finder_table, which silently drop what they can't build a
    row for. Returns (eligible_df, exclusions, as_of_date) where
    exclusions is [{"ticker": str, "reason": str}], one row per excluded
    ticker, and as_of_date is the latest "Last Close Date" across every
    ticker that produced a row at all (the "as-of date of ranking
    scores" DI-01 shows next to Generate).
    """
    universe_tickers = _universe_tickers(universe_key)
    ranked = rank_stocks(goal, universe_key)
    exclusions: List[dict] = []

    if ranked.empty:
        return ranked, [
            {"ticker": t, "reason": "no usable price history from data provider"} for t in universe_tickers
        ], ""

    as_of_date = ranked["Last Close Date"].max()

    produced = set(ranked["Ticker"])
    exclusions += [
        {"ticker": t, "reason": "no usable price history from data provider (fetch failed, or fewer than 70 trading days of history)"}
        for t in universe_tickers
        if t not in produced
    ]

    keep = pd.Series(True, index=ranked.index)

    # (a) no computable score at all for this goal: every weighted input is null.
    weighted_cols = list(GOAL_WEIGHTS[goal].keys())
    all_null = ranked[weighted_cols].isna().all(axis=1)
    exclusions += [
        {"ticker": t, "reason": "no computable score for this goal (all scoring inputs missing)"}
        for t in ranked.loc[all_null, "Ticker"]
    ]
    keep &= ~all_null

    # (b) same data-quality/outlier guard _annualized_return already applies
    # (min_years=2.9, +-200%/-95% sanity bound) -- a data-quality signal,
    # applied regardless of goal, not just when the goal happens to weight
    # this metric.
    insufficient_history = ranked["return_3y_annualized"].isna() & keep
    exclusions += [
        {"ticker": t, "reason": "insufficient price history (under ~2.9 years) or an implausible annualized return, flagged by the outlier guard"}
        for t in ranked.loc[insufficient_history, "Ticker"]
    ]
    keep &= ~insufficient_history

    # (c) stale price: last close more than ~5 trading days behind the
    # universe's own as-of date. Approximated as 9 calendar days (covers
    # weekends without a trading-calendar dependency) -- a pragmatic
    # simplification, not exact trading-day arithmetic.
    as_of_ts = pd.Timestamp(as_of_date)
    stale = ((as_of_ts - pd.to_datetime(ranked["Last Close Date"])).dt.days > 9) & keep
    exclusions += [
        {"ticker": t, "reason": "no price in the last 5 trading days (stale, halted, or delisted)"}
        for t in ranked.loc[stale, "Ticker"]
    ]
    keep &= ~stale

    eligible = ranked.loc[keep].reset_index(drop=True)
    return eligible, exclusions, as_of_date


def _select_sector_picks(sector_df: pd.DataFrame, picks_per_sector: int) -> tuple[pd.DataFrame, Optional[str]]:
    """
    sector_df: one GICS sector's eligible rows, already Score/Market-Cap/
    Ticker sorted descending (rank_stocks's own sort). Greedily takes each
    row unless its Industry is already represented among rows already
    picked for this sector (caps same-sub-industry correlation, e.g. two
    chipmakers, to at most 1 per sector's top-N). If that first pass
    yields fewer than picks_per_sector and eligible rows remain (all from
    already-used industries), a second pass fills the rest ignoring the
    industry cap -- DI-02's "take all if fewer eligible than N" always
    wins over the industry cap; this never returns fewer than
    min(picks_per_sector, len(sector_df)) purely from industry
    concentration.
    """
    if sector_df.empty:
        return sector_df, None

    selected_idx: List[int] = []
    used_industries: set = set()
    for idx, row in sector_df.iterrows():
        if len(selected_idx) >= picks_per_sector:
            break
        if row["Industry"] not in used_industries:
            selected_idx.append(idx)
            used_industries.add(row["Industry"])

    relaxed = False
    if len(selected_idx) < picks_per_sector:
        for idx, _row in sector_df.iterrows():
            if len(selected_idx) >= picks_per_sector:
                break
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            relaxed = True

    note = None
    sector_name = sector_df["GICS Sector"].iloc[0]
    if len(selected_idx) < picks_per_sector:
        note = f"{sector_name}: only {len(selected_idx)} eligible stock{'s' if len(selected_idx) != 1 else ''}"
    elif relaxed:
        note = f"{sector_name}: fewer than {picks_per_sector} distinct sub-industries available, so one industry appears more than once"

    return sector_df.loc[selected_idx], note


def assemble_sector_picks(eligible: pd.DataFrame, picks_per_sector: int) -> tuple[pd.DataFrame, List[str]]:
    """
    Selects picks_per_sector tickers from each of the 11 GICS sectors
    present in `eligible` (DI-02), via _select_sector_picks. Sectors with
    zero eligible tickers are skipped and listed in a note; sectors that
    had to relax the industry cap or take fewer than requested surface
    their own note from _select_sector_picks.
    """
    notes: List[str] = []
    picks_frames = []
    for sector in GICS_SECTORS_ORDER:
        sector_df = eligible[eligible["GICS Sector"] == sector]
        if sector_df.empty:
            continue
        picked, note = _select_sector_picks(sector_df, picks_per_sector)
        picks_frames.append(picked)
        if note:
            notes.append(note)

    present_sectors = set(eligible["GICS Sector"].unique()) if not eligible.empty else set()
    zero_sectors = [s for s in GICS_SECTORS_ORDER if s not in present_sectors]
    if zero_sectors:
        notes.append(f"0 eligible stocks, skipped: {', '.join(zero_sectors)}")

    basket = pd.concat(picks_frames, ignore_index=True) if picks_frames else eligible.iloc[0:0]
    return basket, notes


def build_diversified_basket(
    goal: str, universe_key: str, picks_per_sector: int, max_stocks: Optional[int] = None
) -> pd.DataFrame:
    """
    A custom "index" of individual stocks spread across sectors, instead of
    an existing ETF (see the Fund Screener for that): the picks_per_sector
    highest-Score tickers from each sector present in this universe, using
    the same ranking as /stock-finder. Sector-diversified by construction —
    a hot sector can't dominate the basket just because more of its tickers
    scored well.

    max_stocks, when given, caps the total basket size — useful since
    picks_per_sector alone is a coarse lever (bumping it by 1 adds one
    stock per sector at once, which can overshoot fast in a universe with
    many sectors). The cap is applied by round-robin (see
    _trim_to_max_stocks), not a flat top-N re-sort, so it can't collapse
    the basket back down to one dominant sector.

    Kept for backward compatibility (see GET /diversified-basket, now
    docstring-deprecated in favor of generate_diversified_basket, which
    adds exclusion-reason tracking, sub-industry capping, whole-share
    sizing, and everything else the Diversified Basket page now needs).
    """
    ranked = rank_stocks(goal, universe_key)
    if ranked.empty:
        return ranked

    # ranked is already sorted by Score descending, so a per-group head()
    # keeps each sector's top scorers without re-sorting.
    basket = ranked.groupby("Sector", sort=False, group_keys=False).head(picks_per_sector)
    basket = (
        basket[["Ticker", "Name", "Sector", "Price", "Score"]]
        .sort_values(["Sector", "Score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    if max_stocks is not None and max_stocks > 0 and len(basket) > max_stocks:
        basket, _notes = _trim_to_max_stocks(basket, max_stocks, sector_col="Sector")
    return basket


def _trim_to_max_stocks(
    basket: pd.DataFrame, max_stocks: int, sector_col: str = "GICS Sector"
) -> tuple[pd.DataFrame, List[str]]:
    """
    Round-robins one stock at a time across sectors (round 1 = every
    sector's #1 pick, round 2 = every sector's #2 pick, ...) until
    max_stocks is reached. Within a round, sectors are ordered by the
    Score of the candidate about to be added in THAT round, descending
    (tie-break: sector name ascending) -- not a fixed sector order. This
    means a partial/cutoff round goes to the strongest remaining
    candidates regardless of which sector they're in (DI-04), rather than
    whichever sectors happen to sort first alphabetically always winning
    the last few slots -- concretely: 11 sectors, 2 picks/sector, cap 15
    -> all 11 sector leaders (round 1) plus the 4 highest-scoring #2
    picks (round 2), not the 4 alphabetically-first sectors' #2 picks.
    """
    by_sector: Dict[str, List[int]] = {
        sector: list(group.index) for sector, group in basket.groupby(sector_col, sort=False)
    }
    all_sectors = set(by_sector.keys())
    selected: List[int] = []

    while len(selected) < max_stocks:
        round_candidates = [(sector, queue[0]) for sector, queue in by_sector.items() if queue]
        if not round_candidates:
            break
        round_candidates.sort(key=lambda sc: (-float(basket.loc[sc[1], "Score"]), sc[0]))
        for sector, _idx in round_candidates:
            if len(selected) >= max_stocks:
                break
            selected.append(by_sector[sector].pop(0))

    notes: List[str] = []
    if max_stocks < len(all_sectors):
        included_sectors = {basket.loc[i, sector_col] for i in selected}
        left_out_sectors = sorted(all_sectors - included_sectors)
        if left_out_sectors:
            notes.append(
                f"Max stocks ({max_stocks}) is fewer than the number of sectors — left out entirely: {', '.join(left_out_sectors)}"
            )

    trimmed = basket.loc[selected].sort_values([sector_col, "Score"], ascending=[True, False]).reset_index(drop=True)
    return trimmed, notes


SECTOR_WEIGHTING_MODES = ("equal_dollar", "market_cap_by_sector")


def size_basket_positions(
    basket: pd.DataFrame,
    total_amount: float,
    fractional_shares: bool = False,
    sector_weighting: str = "equal_dollar",
) -> tuple[pd.DataFrame, dict, List[str]]:
    """
    Adds Target $, Shares, Amount, Weight_pct to `basket` (DI-05, and the
    sector-weighting enhancement).

    sector_weighting:
      - "equal_dollar" (default): target per position = total_amount / count.
      - "market_cap_by_sector": each GICS sector's dollar allocation is
        proportional to that sector's aggregate Market Cap ($B) share
        WITHIN THIS BASKET's own selected tickers (the point is weighting
        the picks actually held, not reintroducing the whole universe's
        distribution), split equally across that sector's own picks.

    Whole-share default: Shares = floor(Target $ / Price); leftover cash
    is total_amount minus what actually got invested. fractional_shares=
    True instead rounds Shares to 4 decimal places for an exact equal-
    dollar split.

    Returns (sized_df, totals, warnings). totals = {invested,
    leftover_cash, holding_count}. warnings covers DI-05's "target per
    position is below the price of a selected stock" case under
    whole-share mode, naming every such stock and the three remedies.
    """
    if sector_weighting not in SECTOR_WEIGHTING_MODES:
        raise ValueError(f"sector_weighting must be one of {SECTOR_WEIGHTING_MODES}")
    if basket.empty:
        return basket, {"invested": 0.0, "leftover_cash": total_amount, "holding_count": 0}, []

    basket = basket.copy()
    if sector_weighting == "equal_dollar":
        basket["Target $"] = total_amount / len(basket)
    else:
        sector_cap = basket.groupby("GICS Sector")["Market Cap ($B)"].transform("sum")
        total_cap = basket["Market Cap ($B)"].sum()
        sector_count = basket.groupby("GICS Sector")["Ticker"].transform("count")
        if not total_cap:
            basket["Target $"] = total_amount / len(basket)
        else:
            basket["Target $"] = total_amount * (sector_cap / total_cap) / sector_count

    if fractional_shares:
        basket["Shares"] = (basket["Target $"] / basket["Price"]).round(4)
    else:
        basket["Shares"] = np.floor(basket["Target $"] / basket["Price"])

    basket["Amount"] = basket["Shares"] * basket["Price"]
    invested = float(basket["Amount"].sum())
    leftover_cash = round(total_amount - invested, 2)
    basket["Weight_pct"] = (basket["Amount"] / invested * 100) if invested else 0.0

    warnings: List[str] = []
    if not fractional_shares:
        zero_share_rows = basket[basket["Shares"] <= 0]
        if not zero_share_rows.empty:
            names = ", ".join(f"{r.Ticker} (${r.Price:.2f})" for r in zero_share_rows.itertuples())
            warnings.append(
                f"Target per position is below the price of: {names}. "
                f"Enable fractional shares, raise the total amount, or lower Max stocks."
            )

    totals = {
        "invested": invested,
        "leftover_cash": leftover_cash,
        "holding_count": int((basket["Shares"] > 0).sum()),
    }
    return basket, totals, warnings


@ttl_cache(maxsize=4, ttl_seconds=3600)
def compute_sp500_sector_mix() -> Dict[str, float]:
    """
    Approximates "SPY's sector mix" as the aggregate market-cap share by
    GICS sector across the full S&P 500 scan -- this app has no source
    for SPY's real holdings/weights data. Every caller must label this as
    an approximation, not real index data.
    """
    df = get_stock_finder_table(SP500_UNIVERSE_NAME).copy()
    if df.empty:
        return {}
    df["GICS Sector"] = df["Sector"].map(_gics_sector)
    by_sector = df.groupby("GICS Sector")["Market Cap ($B)"].sum(min_count=1).dropna()
    total = by_sector.sum()
    if not total:
        return {}
    return (by_sector / total * 100).round(2).to_dict()


def check_concentration_warning(basket_sector_weights: Dict[str, float], universe_sector_count: int) -> Optional[str]:
    """DI-09: warn (non-blocking) when a universe can't produce real
    sector spread -- fewer than 5 sectors present, or any single sector
    would take more than 40% of the basket."""
    if universe_sector_count < 5:
        return f"This universe only spans {universe_sector_count} sector(s); the basket will be less diversified."
    if basket_sector_weights:
        top_sector = max(basket_sector_weights, key=basket_sector_weights.get)
        if basket_sector_weights[top_sector] > 40.0:
            return f"This universe is concentrated in {top_sector}; the basket will be less diversified."
    return None


def get_universe_sector_preview(universe_key: str) -> dict:
    """Cheap, cache-backed (reuses get_stock_finder_table's own TTL cache):
    per-universe stock count + sector counts + as-of date, for DI-01's
    per-universe description and DI-09's pre-generation concentration
    check -- both render before the user clicks Generate, no extra
    network cost beyond what's already cached."""
    df = get_stock_finder_table(universe_key)
    if df.empty:
        return {"stock_count": 0, "sector_counts": {}, "as_of_date": None}
    df = df.copy()
    df["GICS Sector"] = df["Sector"].map(_gics_sector)
    counts = df["GICS Sector"].value_counts().to_dict()
    as_of = df["Last Close Date"].max() if "Last Close Date" in df else None
    return {"stock_count": len(df), "sector_counts": counts, "as_of_date": as_of}


def compute_basket_risk_preview(basket: pd.DataFrame, lookback: str = "1y") -> dict:
    """
    Approximates the basket's risk profile AS IF today's picks/weights
    had been held, unchanged, for the whole lookback window -- a
    retroactive application of today's membership to history, not a real
    trade-by-trade backtest. Callers must disclose this.
    """
    if basket.empty:
        return {
            "annualized_volatility_pct": None, "beta_to_spy": None, "max_drawdown_pct": None,
            "largest_single_stock_weight_pct": None, "largest_single_sector_weight_pct": None,
            "lookback": lookback, "excluded_from_risk": [],
        }

    closes: Dict[str, pd.Series] = {}
    for ticker in basket["Ticker"]:
        hist = get_cached_history(ticker, lookback, auto_adjust=True)
        if not hist.empty:
            closes[ticker] = hist["Close"]
    spy_hist = get_cached_history("SPY", lookback, auto_adjust=True)
    spy_close = spy_hist["Close"] if not spy_hist.empty else pd.Series(dtype=float)

    excluded_from_risk = [t for t in basket["Ticker"] if t not in closes]
    if not closes or spy_close.empty:
        return {
            "annualized_volatility_pct": None, "beta_to_spy": None, "max_drawdown_pct": None,
            "largest_single_stock_weight_pct": float(basket["Weight_pct"].max()) if "Weight_pct" in basket else None,
            "largest_single_sector_weight_pct": (
                float(basket.groupby("GICS Sector")["Weight_pct"].sum().max()) if "Weight_pct" in basket else None
            ),
            "lookback": lookback, "excluded_from_risk": excluded_from_risk,
        }

    price_df = pd.DataFrame({**closes, "__SPY__": spy_close}).dropna(how="any")
    # A ticker with too little overlap against the common date range is
    # dropped from THIS calculation only (not from the basket itself) --
    # its dollar weight is renormalized across the remaining tickers.
    kept_tickers = [t for t in closes if t in price_df.columns]
    excluded_from_risk += [t for t in basket["Ticker"] if t not in kept_tickers and t not in excluded_from_risk]

    returns_df = price_df[kept_tickers].pct_change().dropna()
    spy_returns = price_df["__SPY__"].pct_change().dropna()
    returns_df, spy_returns = returns_df.align(spy_returns, join="inner", axis=0)

    weights = basket.set_index("Ticker").loc[kept_tickers, "Weight_pct"] / 100.0
    weights = weights / weights.sum() if weights.sum() else weights

    basket_returns = (returns_df * weights).sum(axis=1)

    annualized_volatility_pct = float(basket_returns.std() * np.sqrt(252) * 100) if not basket_returns.empty else None
    if basket_returns.var() and spy_returns.loc[basket_returns.index].var():
        cov = np.cov(basket_returns, spy_returns.loc[basket_returns.index], ddof=1)
        beta_to_spy = float(cov[0, 1] / cov[1, 1])
    else:
        beta_to_spy = None
    drawdown = max_drawdown_pct((basket_returns * 100).tolist())

    return {
        "annualized_volatility_pct": annualized_volatility_pct,
        "beta_to_spy": beta_to_spy,
        "max_drawdown_pct": drawdown,
        "largest_single_stock_weight_pct": float(basket["Weight_pct"].max()),
        "largest_single_sector_weight_pct": float(basket.groupby("GICS Sector")["Weight_pct"].sum().max()),
        "lookback": lookback,
        "excluded_from_risk": excluded_from_risk,
    }


def replace_basket_ticker(eligible: pd.DataFrame, current_tickers: List[str], removed_ticker: str) -> Optional[dict]:
    """DI-06: when the user removes a ticker, returns the next-ranked
    eligible ticker's row from the SAME GICS sector that isn't already in
    the basket, or None if that sector has nothing left."""
    removed_rows = eligible[eligible["Ticker"] == removed_ticker]
    if removed_rows.empty:
        return None
    sector = removed_rows.iloc[0]["GICS Sector"]
    candidates = eligible[
        (eligible["GICS Sector"] == sector) & (~eligible["Ticker"].isin(current_tickers))
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0].to_dict()


def generate_diversified_basket(
    goal: str,
    universe_key: str,
    picks_per_sector: int,
    max_stocks: Optional[int],
    total_amount: float,
    fractional_shares: bool = False,
    sector_weighting: str = "equal_dollar",
    excluded_tickers: Optional[List[str]] = None,
) -> dict:
    """
    Full DI-01-DI-09 + risk-preview orchestration: eligibility/exclusions
    -> per-sector selection (sub-industry capped) -> max-stocks trim ->
    dollar sizing -> concentration warning -> risk preview. excluded_
    tickers lets the caller regenerate with specific tickers removed
    (DI-06's remove flow), without needing separate client-side logic
    that could drift from this exact algorithm.
    """
    eligible, exclusions, as_of_date = get_basket_candidates(goal, universe_key)
    if excluded_tickers:
        eligible = eligible[~eligible["Ticker"].isin(excluded_tickers)].reset_index(drop=True)

    if eligible.empty:
        return {
            "as_of_date": as_of_date, "holdings": [], "sector_summary": [], "excluded": exclusions,
            "sector_notes": [], "trim_notes": [], "totals": {"invested": 0.0, "leftover_cash": total_amount, "holding_count": 0},
            "warnings": [], "concentration_warning": None, "risk_preview": compute_basket_risk_preview(eligible),
        }

    basket, sector_notes = assemble_sector_picks(eligible, picks_per_sector)

    trim_notes: List[str] = []
    if max_stocks is not None and max_stocks > 0 and len(basket) > max_stocks:
        basket, trim_notes = _trim_to_max_stocks(basket, max_stocks, sector_col="GICS Sector")

    sized, totals, warnings = size_basket_positions(basket, total_amount, fractional_shares, sector_weighting)

    spy_mix = compute_sp500_sector_mix()
    sector_summary = []
    basket_sector_weights: Dict[str, float] = {}
    if not sized.empty:
        grouped = sized.groupby("GICS Sector").agg(Count=("Ticker", "count"), Weight_pct=("Weight_pct", "sum"))
        for sector, row in grouped.iterrows():
            basket_sector_weights[sector] = float(row["Weight_pct"])
            sector_summary.append({
                "Sector": sector, "Count": int(row["Count"]), "Weight_pct": round(float(row["Weight_pct"]), 2),
                "Spy_Approx_Weight_pct": round(spy_mix.get(sector, 0.0), 2),
            })

    universe_preview = get_universe_sector_preview(universe_key)
    concentration_warning = check_concentration_warning(
        basket_sector_weights, len(universe_preview.get("sector_counts", {}))
    )

    risk_preview = compute_basket_risk_preview(sized)

    holdings_cols = ["Ticker", "Name", "GICS Sector", "Industry", "Score", "Price", "Shares", "Amount", "Weight_pct"]
    holdings = sized[holdings_cols].round({"Score": 1, "Price": 2, "Shares": 4, "Amount": 2, "Weight_pct": 2}).to_dict("records")

    return {
        "as_of_date": as_of_date,
        "holdings": holdings,
        "sector_summary": sector_summary,
        "excluded": exclusions,
        "sector_notes": sector_notes,
        "trim_notes": trim_notes,
        "totals": totals,
        "warnings": warnings,
        "concentration_warning": concentration_warning,
        "risk_preview": risk_preview,
    }


def score_stock_ticker(goal: str, ticker_symbol: str) -> pd.DataFrame:
    df = get_single_stock_table(ticker_symbol).copy()
    if df.empty:
        return df

    df["return_1m"] = df["1M Return %"]
    df["return_3m"] = df["3M Return %"]
    df["return_6m"] = df["6M Return %"]
    df["return_1y"] = df["1Y Return %"]
    df["return_3y_annualized"] = df["3Y Annualized %"]
    df["rsi_balance"] = df["RSI Balance"]
    df["macd_signal_strength"] = df["MACD Strength"]
    df["volume_strength"] = df["Volume Strength %"]
    df["volatility_6m"] = df["6M Volatility %"]
    df["max_drawdown_1y"] = df["1Y Max Drawdown %"]
    df["max_drawdown_3y"] = df["3Y Max Drawdown %"]
    df["sharpe_3y"] = df["3Y Sharpe"]
    df["forward_pe"] = df["Forward PE"]
    df["revenue_growth"] = df["Revenue Growth %"]
    df["earnings_growth"] = df["Earnings Growth %"]

    weights = GOAL_WEIGHTS[goal]
    total_score = pd.Series([0.0] * len(df), index=df.index)
    for metric, weight in weights.items():
        total_score += _score_series(df[metric], lower_is_better=metric in LOWER_IS_BETTER) * weight

    df["Score"] = (total_score * 100).round(1)
    return df.reset_index(drop=True)
