"""
I/O layer for Phase 1's two-score system (docs/stock-analysis-
requirements.html, SCR-1..4): fetches each factor's raw per-ticker input,
preferring the PIT store and falling back to a live yfinance fetch only
when the PIT store doesn't have enough history for that ticker yet --
the same hybrid pattern services/signal_publication_service.py's
build_daily_signal_set_hybrid already established for the momentum-
ranking publication (score_tickers_from_pit/merge_pit_and_live_scores),
extended here to RSI and trailing volatility, which have no PIT-native
helper yet.

Every fetch function tags each ticker's factor with source: "pit"|"live"
-- the same honesty convention /track-record's own data-source badge
already uses -- so callers never have to guess where a number came from.
As pit_prices deepens past each factor's threshold below, live-fallback
usage shrinks to zero on its own, with no code change.

Earnings surprise and earnings revisions are the one exception to that
"live means shallow PIT history, will self-heal" reading: both are
always source: "live", permanently -- not a depth fallback. Both are
inherently quarterly/rolling-analyst-consensus data, not a daily price
series, so a dedicated PIT table would add storage complexity without a
real benefit; they're just re-fetched each run via yfinance_cache's own
15-minute TTL cache.

All DB access goes through service_conn (cross-user, public PIT data --
see web/backend/paper_order_sync.py for the same pattern in this repo).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from services.pit_signal_service import score_tickers_from_pit
from services.stock_finder_service import _gics_sector, _universe_tickers
from services.stock_score_service import (
    LONG_TERM_WEIGHTS,
    SHORT_TERM_WEIGHTS,
    compute_factor_contributions,
    compute_long_score,
    compute_short_score,
    derive_confidence,
    flip_count_from_signal_history,
    percentile_rank,
    score_to_signal,
    sector_percentile,
    sector_rank,
)
from services.yfinance_cache import get_cached_earnings_dates, get_cached_eps_trend, get_cached_history, get_cached_info
from web.backend.db import service_conn
from web.backend.pit_prices import _eastern_today

logger = logging.getLogger(__name__)

MOMENTUM_LOOKBACK_DAYS = 30
RSI_MIN_ROWS = 15
# Only trust PIT for volatility once a full year is on record; below that,
# live fallback -- unlike momentum/RSI, a volatility estimate from a
# handful of weeks is unreliable enough not to bother treating as PIT.
VOLATILITY_TRADING_DAYS = 252


async def _fetch_pit_price_history(tickers: list[str], as_of_date_: date, min_days_back: int) -> dict[str, list[dict]]:
    """Every PIT price row for these tickers up to as_of_date_, grouped by
    ticker, oldest first -- shared by every factor below so pit_prices is
    hit once per capture run, not once per factor."""
    if not tickers:
        return {}
    window_start = as_of_date_ - timedelta(days=int(min_days_back * 1.6) + 10)
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, price_date, close
            FROM pit_prices
            WHERE ticker = ANY($1::text[]) AND price_date BETWEEN $2 AND $3
            ORDER BY ticker, price_date
            """,
            tickers, window_start, as_of_date_,
        )
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append({"price_date": r["price_date"], "close": r["close"]})
    return by_ticker


def _compute_rsi(closes: list[float]) -> Optional[float]:
    """Same RSI(14) formula as services/technical_service.py's
    add_indicators (delta -> rolling(14) gain/loss mean -> RS -> RSI),
    inlined rather than calling that function directly: add_indicators
    also computes SMA50/Bollinger and drops every row missing ANY
    indicator, which would demand 50+ rows just to read a single RSI
    value -- far stricter than RSI's own real 14-period requirement."""
    if len(closes) < RSI_MIN_ROWS:
        return None
    close = pd.Series(closes)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + gain / loss))
    rsi = rsi_series.iloc[-1]
    return round(float(rsi), 2) if pd.notna(rsi) else None


def _annualized_volatility(closes: pd.Series) -> Optional[float]:
    """Same std*sqrt(252)*100 formula as stock_finder_service.
    compute_basket_risk_preview uses on a weighted basket, applied here to
    one ticker's own daily returns, unweighted."""
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    return round(float(returns.std() * (252 ** 0.5) * 100), 2)


def _live_momentum(ticker: str) -> dict:
    hist = get_cached_history(ticker, "3mo", auto_adjust=True)
    close = hist["Close"].dropna() if not hist.empty else hist
    if close.empty or len(close) < MOMENTUM_LOOKBACK_DAYS + 1:
        return {"raw": None, "source": "live"}
    start, end = float(close.iloc[-(MOMENTUM_LOOKBACK_DAYS + 1)]), float(close.iloc[-1])
    if not start:
        return {"raw": None, "source": "live"}
    return {"raw": round((end / start - 1.0) * 100, 4), "source": "live"}


def _live_reversal(ticker: str) -> dict:
    hist = get_cached_history(ticker, "3mo", auto_adjust=True)
    if hist.empty:
        return {"raw": None, "source": "live"}
    return {"raw": _compute_rsi(hist["Close"].dropna().tolist()), "source": "live"}


def _live_volatility(ticker: str) -> dict:
    hist = get_cached_history(ticker, "1y", auto_adjust=True)
    if hist.empty:
        return {"raw": None, "source": "live"}
    return {"raw": _annualized_volatility(hist["Close"].dropna()), "source": "live"}


def _momentum_and_reversal_from_rows(
    tickers: list[str],
    pit_rows_by_ticker: dict[str, list[dict]],
    live_momentum_fn=None,
    live_reversal_fn=None,
) -> dict[str, dict]:
    """Pure hybrid decision logic, given already-fetched PIT rows and
    live-fetch callables -- separated from the DB fetch itself (see
    fetch_momentum_and_reversal_inputs below) so it's directly
    unit-testable with synthetic PIT rows and fake live-fetch functions,
    mirroring web/backend/plaid_sync.py's split between a DB-touching
    wrapper and its testable reconciliation function. Each factor falls
    back independently -- a ticker can clear the momentum threshold
    (lookback_days+1 rows) without clearing RSI's (15 rows), or vice
    versa, so one factor being PIT-sourced never implies the other is."""
    live_momentum_fn = live_momentum_fn or _live_momentum
    live_reversal_fn = live_reversal_fn or _live_reversal

    flat_pit_rows = [
        {"ticker": t, "price_date": r["price_date"], "close": r["close"]}
        for t, rows in pit_rows_by_ticker.items() for r in rows
    ]
    pit_momentum = {r["ticker"]: r["trailing_return_pct"] for r in score_tickers_from_pit(flat_pit_rows, MOMENTUM_LOOKBACK_DAYS)}

    result: dict[str, dict] = {}
    for ticker in tickers:
        momentum = (
            {"raw": pit_momentum[ticker], "source": "pit"} if ticker in pit_momentum else live_momentum_fn(ticker)
        )

        pit_closes = [r["close"] for r in pit_rows_by_ticker.get(ticker, [])]
        if len(pit_closes) >= RSI_MIN_ROWS:
            rsi = _compute_rsi(pit_closes)
            reversal = {"raw": rsi, "source": "pit"} if rsi is not None else live_reversal_fn(ticker)
        else:
            reversal = live_reversal_fn(ticker)

        result[ticker] = {"momentum": momentum, "reversal": reversal}
    return result


async def fetch_momentum_and_reversal_inputs(tickers: list[str], as_of_date_: date) -> dict[str, dict]:
    """Per ticker: {"momentum": {"raw": pct_return, "source": "pit"|"live"},
    "reversal": {"raw": rsi, "source": "pit"|"live"}}. Fetches PIT rows,
    then delegates to the pure, separately-tested merge logic above."""
    pit_rows_by_ticker = await _fetch_pit_price_history(tickers, as_of_date_, MOMENTUM_LOOKBACK_DAYS)
    return _momentum_and_reversal_from_rows(tickers, pit_rows_by_ticker)


def _volatility_from_rows(tickers: list[str], pit_rows_by_ticker: dict[str, list[dict]], live_volatility_fn=None) -> dict[str, dict]:
    """Pure hybrid decision logic, same split rationale as
    _momentum_and_reversal_from_rows above."""
    live_volatility_fn = live_volatility_fn or _live_volatility
    result: dict[str, dict] = {}
    for ticker in tickers:
        closes = [r["close"] for r in pit_rows_by_ticker.get(ticker, [])]
        if len(closes) >= VOLATILITY_TRADING_DAYS:
            vol = _annualized_volatility(pd.Series(closes))
            result[ticker] = {"raw": vol, "source": "pit"} if vol is not None else live_volatility_fn(ticker)
        else:
            result[ticker] = live_volatility_fn(ticker)
    return result


async def fetch_low_volatility_inputs(tickers: list[str], as_of_date_: date) -> dict[str, dict]:
    """Per ticker: {"raw": annualized_volatility_pct, "source": "pit"|"live"}."""
    pit_rows_by_ticker = await _fetch_pit_price_history(tickers, as_of_date_, VOLATILITY_TRADING_DAYS)
    return _volatility_from_rows(tickers, pit_rows_by_ticker)


async def fetch_value_growth_and_quality_inputs(tickers: list[str], as_of_date_: date) -> dict[str, dict]:
    """Per ticker: {"value": {"raw": forward_pe}, "growth": {"raw_revenue":
    ..., "raw_earnings": ...}, "quality": {"raw_roe": ..., "raw_margin":
    ...}}, all sourced from the latest pit_fundamentals row on or before
    as_of_date_ -- no depth problem, only the latest snapshot matters
    (pit_fundamentals has had one row/ticker/day since capture began)."""
    if not tickers:
        return {}
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, forward_pe, revenue_growth_pct, earnings_growth_pct,
                   return_on_equity_pct, profit_margin_pct, sector
            FROM pit_fundamentals
            WHERE ticker = ANY($1::text[]) AND as_of_date <= $2
            ORDER BY ticker, as_of_date DESC
            """,
            tickers, as_of_date_,
        )
    by_ticker = {r["ticker"]: r for r in rows}
    result: dict[str, dict] = {}
    for ticker in tickers:
        row = by_ticker.get(ticker)
        result[ticker] = {
            "value": {"raw": row["forward_pe"] if row else None, "source": "pit"},
            "growth": {
                "raw_revenue": row["revenue_growth_pct"] if row else None,
                "raw_earnings": row["earnings_growth_pct"] if row else None,
                "source": "pit",
            },
            "quality": {
                "raw_roe": row["return_on_equity_pct"] if row else None,
                "raw_margin": row["profit_margin_pct"] if row else None,
                "source": "pit",
            },
        }
    return result


def blend_growth(revenue_growth_pct: Optional[float], earnings_growth_pct: Optional[float]) -> Optional[float]:
    """One raw growth value to percentile-rank, from whichever of
    revenue/earnings growth is available (averaged when both are)."""
    values = [v for v in (revenue_growth_pct, earnings_growth_pct) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def blend_quality(roe_pct: Optional[float], profit_margin_pct: Optional[float]) -> Optional[float]:
    """One raw quality value to percentile-rank, mirroring blend_growth
    exactly: average of whichever of ROE%/profit-margin% is available.
    Debt-to-equity is deliberately excluded -- it's lower-is-better, and
    averaging it in here pre-percentile with these two higher-is-better
    metrics would distort the blend; that needs a percentile-then-blend
    design instead of this raw-then-percentile one, out of scope here."""
    values = [v for v in (roe_pct, profit_margin_pct) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _latest_eps_surprise(earnings_df: pd.DataFrame) -> Optional[float]:
    """Pure: the most recently REPORTED quarter's Surprise(%) from
    get_cached_earnings_dates' frame -- skips upcoming (NaN Reported EPS)
    rows. Confirmed by hand-calculation against real AAPL/MSFT/NVDA data
    that yfinance's Surprise(%) is already a plain percent (positive =
    beat), not a fraction -- see services/stock_detail_service.py's
    past_earnings_dates, which omits this same column because that
    confirmation hadn't happened yet when it was written."""
    if earnings_df.empty or "Surprise(%)" not in earnings_df.columns or "Reported EPS" not in earnings_df.columns:
        return None
    reported = earnings_df.dropna(subset=["Reported EPS"])
    if reported.empty:
        return None
    surprise = reported.sort_index(ascending=False).iloc[0].get("Surprise(%)")
    return round(float(surprise), 2) if pd.notna(surprise) else None


def _live_earnings_surprise(ticker: str) -> dict:
    return {"raw": _latest_eps_surprise(get_cached_earnings_dates(ticker)), "source": "live"}


async def fetch_earnings_surprise_inputs(tickers: list[str]) -> dict[str, dict]:
    """Per ticker: {"raw": surprise_pct, "source": "live"} -- always live,
    see the module docstring for why this factor is never PIT-stored."""
    results = await asyncio.gather(*[asyncio.to_thread(_live_earnings_surprise, t) for t in tickers])
    return dict(zip(tickers, results))


def _eps_revision_from_trend(eps_trend_df: pd.DataFrame) -> Optional[float]:
    """Pure: (current - 30daysAgo) / abs(30daysAgo) * 100 from the "0q"
    (current-quarter, the nearest-term and most decision-relevant) row of
    get_cached_eps_trend's frame."""
    if eps_trend_df.empty or "0q" not in eps_trend_df.index:
        return None
    row = eps_trend_df.loc["0q"]
    current, month_ago = row.get("current"), row.get("30daysAgo")
    if current is None or month_ago is None or pd.isna(current) or pd.isna(month_ago) or month_ago == 0:
        return None
    return round(float((current - month_ago) / abs(month_ago) * 100), 2)


def _live_earnings_revisions(ticker: str) -> dict:
    return {"raw": _eps_revision_from_trend(get_cached_eps_trend(ticker)), "source": "live"}


async def fetch_earnings_revisions_inputs(tickers: list[str]) -> dict[str, dict]:
    """Per ticker: {"raw": revision_pct, "source": "live"} -- always live,
    same reasoning as fetch_earnings_surprise_inputs above."""
    results = await asyncio.gather(*[asyncio.to_thread(_live_earnings_revisions, t) for t in tickers])
    return dict(zip(tickers, results))


async def resolve_sector_map(tickers: list[str], as_of_date_: date) -> dict[str, str]:
    """GICS-normalized sector per ticker (services.stock_finder_service.
    _gics_sector), preferring the sector already captured alongside that
    day's pit_fundamentals row (added specifically so this doesn't need
    its own full-universe .info pass) and falling back to a live,
    already-shared-cache .info call only for a ticker missing that."""
    if not tickers:
        return {}
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (ticker) ticker, sector
            FROM pit_fundamentals
            WHERE ticker = ANY($1::text[]) AND as_of_date <= $2 AND sector IS NOT NULL
            ORDER BY ticker, as_of_date DESC
            """,
            tickers, as_of_date_,
        )
    by_ticker = {r["ticker"]: r["sector"] for r in rows}
    result: dict[str, str] = {}
    for ticker in tickers:
        raw_sector = by_ticker.get(ticker)
        if raw_sector is None:
            raw_sector = get_cached_info(ticker).get("sector")
        result[ticker] = _gics_sector(raw_sector)
    return result


def resolve_universe_tickers(universe_id: str) -> list[str]:
    """Thin re-export so callers of this module never import
    stock_finder_service directly for just this one lookup."""
    return list(_universe_tickers(universe_id))


async def _fetch_prior_signal_history(tickers: list[str], universe_id: str, before_date: date) -> dict[str, list]:
    """Every (as_of_date, short_signal, long_signal) row already on record
    for these tickers, strictly before before_date -- feeds
    flip_count_from_signal_history once today's own just-computed signal
    is appended to it (see compute_and_persist_daily_scores)."""
    if not tickers:
        return {}
    async with service_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, as_of_date, short_signal, long_signal
            FROM stock_scores
            WHERE ticker = ANY($1::text[]) AND universe_id = $2 AND as_of_date < $3
            ORDER BY ticker, as_of_date
            """,
            tickers, universe_id, before_date,
        )
    by_ticker: dict[str, list] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append((r["as_of_date"], r["short_signal"], r["long_signal"]))
    return by_ticker


def _factor_contribution(contributions: list[dict], factor: str) -> Optional[float]:
    return next((c["contribution"] for c in contributions if c["factor"] == factor), None)


async def compute_and_persist_daily_scores(universe_id: str = "All", as_of_date_: Optional[date] = None) -> int:
    """Orchestrates every factor fetch above + the pure functions in
    services/stock_score_service.py into one row per ticker, inserted
    into stock_scores. The function the scheduler's nightly job calls
    (see web/backend/scheduler.py's _compute_stock_scores_job). Returns
    the number of rows actually inserted (new tickers only -- ON CONFLICT
    DO NOTHING makes this safely re-runnable, same guarantee as every
    other PIT-family capture in this app)."""
    if as_of_date_ is None:
        # Same ET-not-server-UTC labeling web/backend/pit_prices.py's
        # _eastern_today() exists to fix -- a naive date.today() call in
        # the evening would mislabel today's scores with tomorrow's date.
        as_of_date_ = _eastern_today()

    tickers = resolve_universe_tickers(universe_id)
    if not tickers:
        return 0

    (
        momentum_reversal, volatility, value_growth_quality, sector_map, prior_history,
        earnings_surprise, earnings_revisions,
    ) = await asyncio.gather(
        fetch_momentum_and_reversal_inputs(tickers, as_of_date_),
        fetch_low_volatility_inputs(tickers, as_of_date_),
        fetch_value_growth_and_quality_inputs(tickers, as_of_date_),
        resolve_sector_map(tickers, as_of_date_),
        _fetch_prior_signal_history(tickers, universe_id, as_of_date_),
        fetch_earnings_surprise_inputs(tickers),
        fetch_earnings_revisions_inputs(tickers),
    )

    momentum_raw = {t: momentum_reversal[t]["momentum"]["raw"] for t in tickers}
    reversal_raw = {t: momentum_reversal[t]["reversal"]["raw"] for t in tickers}
    value_raw = {t: value_growth_quality[t]["value"]["raw"] for t in tickers}
    growth_raw = {
        t: blend_growth(value_growth_quality[t]["growth"]["raw_revenue"], value_growth_quality[t]["growth"]["raw_earnings"])
        for t in tickers
    }
    low_vol_raw = {t: volatility[t]["raw"] for t in tickers}
    quality_raw = {
        t: blend_quality(value_growth_quality[t]["quality"]["raw_roe"], value_growth_quality[t]["quality"]["raw_margin"])
        for t in tickers
    }
    earnings_surprise_raw = {t: earnings_surprise[t]["raw"] for t in tickers}
    earnings_revisions_raw = {t: earnings_revisions[t]["raw"] for t in tickers}

    # Reversal/value/low-vol are all "lower raw value scores higher"
    # (oversold RSI, cheap P/E, calm volatility). Everything else here --
    # momentum, growth, quality, earnings surprise, earnings revisions --
    # is "higher raw value scores higher" (a bigger beat/upward revision
    # is better, same as more growth).
    momentum_pct = percentile_rank(momentum_raw)
    reversal_pct = percentile_rank(reversal_raw, lower_is_better=True)
    value_pct = percentile_rank(value_raw, lower_is_better=True)
    growth_pct = percentile_rank(growth_raw)
    low_vol_pct = percentile_rank(low_vol_raw, lower_is_better=True)
    quality_pct = percentile_rank(quality_raw)
    earnings_surprise_pct = percentile_rank(earnings_surprise_raw)
    earnings_revisions_pct = percentile_rank(earnings_revisions_raw)

    short_scores = compute_short_score(momentum_pct, reversal_pct, earnings_surprise_pct, earnings_revisions_pct)
    long_scores = compute_long_score(value_pct, growth_pct, low_vol_pct, quality_pct)
    short_sector_pct = sector_percentile(short_scores, sector_map)
    long_sector_pct = sector_percentile(long_scores, sector_map)
    # SCR-3: the composite score re-ranked against the whole universe --
    # NOT the same number as the weighted-average-of-factor-percentiles
    # that produced the score itself, since a weighted average of
    # percentiles isn't guaranteed to itself be a valid percentile of the
    # resulting composite. percentile_rank with no sector grouping IS a
    # genuine universe-wide percentile, by construction.
    short_universe_pct = percentile_rank(short_scores)
    long_universe_pct = percentile_rank(long_scores)
    short_sec_rank = sector_rank(short_scores, sector_map)
    long_sec_rank = sector_rank(long_scores, sector_map)

    inserted = 0
    async with service_conn() as conn:
        for ticker in tickers:
            short_score = short_scores.get(ticker)
            long_score = long_scores.get(ticker)
            if short_score is None and long_score is None:
                # Nothing computable at all for this ticker today (no PIT
                # or live data from any factor) -- skip rather than write
                # a row of fabricated zeros.
                continue

            short_signal = score_to_signal(short_score)
            long_signal = score_to_signal(long_score)

            history = prior_history.get(ticker, [])
            short_stability = flip_count_from_signal_history(
                [(d, s) for d, s, _ in history] + [(as_of_date_, short_signal)]
            )
            long_stability = flip_count_from_signal_history(
                [(d, l) for d, _, l in history] + [(as_of_date_, long_signal)]
            )
            short_confidence = derive_confidence(short_stability)
            long_confidence = derive_confidence(long_stability)

            short_contributions = compute_factor_contributions(
                {
                    "momentum": momentum_raw[ticker], "reversal": reversal_raw[ticker],
                    "earnings_surprise": earnings_surprise_raw[ticker],
                    "earnings_revisions": earnings_revisions_raw[ticker],
                },
                {
                    "momentum": momentum_pct[ticker], "reversal": reversal_pct[ticker],
                    "earnings_surprise": earnings_surprise_pct[ticker],
                    "earnings_revisions": earnings_revisions_pct[ticker],
                },
                SHORT_TERM_WEIGHTS,
            )
            long_contributions = compute_factor_contributions(
                {
                    "value": value_raw[ticker], "growth": growth_raw[ticker], "low_vol": low_vol_raw[ticker],
                    "quality": quality_raw[ticker],
                },
                {
                    "value": value_pct[ticker], "growth": growth_pct[ticker], "low_vol": low_vol_pct[ticker],
                    "quality": quality_pct[ticker],
                },
                LONG_TERM_WEIGHTS,
            )

            factor_detail = {
                "momentum": {
                    **momentum_reversal[ticker]["momentum"],
                    "percentile": momentum_pct[ticker],
                    "contribution": _factor_contribution(short_contributions, "momentum"),
                },
                "reversal": {
                    **momentum_reversal[ticker]["reversal"],
                    "percentile": reversal_pct[ticker],
                    "contribution": _factor_contribution(short_contributions, "reversal"),
                },
                "earnings_surprise": {
                    **earnings_surprise[ticker],
                    "percentile": earnings_surprise_pct[ticker],
                    "contribution": _factor_contribution(short_contributions, "earnings_surprise"),
                },
                "earnings_revisions": {
                    **earnings_revisions[ticker],
                    "percentile": earnings_revisions_pct[ticker],
                    "contribution": _factor_contribution(short_contributions, "earnings_revisions"),
                },
                "value": {
                    **value_growth_quality[ticker]["value"],
                    "percentile": value_pct[ticker],
                    "contribution": _factor_contribution(long_contributions, "value"),
                },
                "growth": {
                    "raw": growth_raw[ticker],
                    "raw_revenue": value_growth_quality[ticker]["growth"]["raw_revenue"],
                    "raw_earnings": value_growth_quality[ticker]["growth"]["raw_earnings"],
                    "source": value_growth_quality[ticker]["growth"]["source"],
                    "percentile": growth_pct[ticker],
                    "contribution": _factor_contribution(long_contributions, "growth"),
                },
                "low_vol": {
                    **volatility[ticker],
                    "percentile": low_vol_pct[ticker],
                    "contribution": _factor_contribution(long_contributions, "low_vol"),
                },
                "quality": {
                    "raw": quality_raw[ticker],
                    "raw_roe": value_growth_quality[ticker]["quality"]["raw_roe"],
                    "raw_margin": value_growth_quality[ticker]["quality"]["raw_margin"],
                    "source": value_growth_quality[ticker]["quality"]["source"],
                    "percentile": quality_pct[ticker],
                    "contribution": _factor_contribution(long_contributions, "quality"),
                },
            }

            result = await conn.execute(
                """
                INSERT INTO stock_scores (
                    as_of_date, universe_id, ticker,
                    short_score, short_signal, short_confidence_score, short_confidence_label,
                    long_score, long_signal, long_confidence_score, long_confidence_label,
                    sector_key, short_sector_percentile, long_sector_percentile, factor_detail,
                    short_universe_percentile, long_universe_percentile,
                    short_sector_rank, short_sector_count, long_sector_rank, long_sector_count
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16,$17,$18,$19,$20,$21)
                ON CONFLICT (as_of_date, universe_id, ticker) DO NOTHING
                """,
                as_of_date_, universe_id, ticker,
                short_score, short_signal, short_confidence["score"], short_confidence["label"],
                long_score, long_signal, long_confidence["score"], long_confidence["label"],
                sector_map.get(ticker, "Unknown"),
                short_sector_pct.get(ticker), long_sector_pct.get(ticker),
                json.dumps(factor_detail, default=str),
                short_universe_pct.get(ticker), long_universe_pct.get(ticker),
                (short_sec_rank.get(ticker) or {}).get("rank"), (short_sec_rank.get(ticker) or {}).get("of"),
                (long_sec_rank.get(ticker) or {}).get("rank"), (long_sec_rank.get(ticker) or {}).get("of"),
            )
            if result == "INSERT 0 1":
                inserted += 1

    logger.info("Stock score capture for %s: %d/%d tickers newly inserted", universe_id, inserted, len(tickers))
    return inserted
