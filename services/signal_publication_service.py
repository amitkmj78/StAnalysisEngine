import datetime
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf

from .data_service import get_latest_price
from .pit_signal_service import merge_pit_and_live_scores, score_tickers_from_pit
from .portfolio_compare_service import derive_confidence
from .prediction_service import generate_trading_signal, predict_future_prices
from .ranking_utils import rank_tickers_against_universe
from .stock_finder_service import _universe_tickers, get_stock_finder_table
from .stock_score_service import flip_count_from_signal_history
from .yfinance_cache import get_cached_history

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TOP_N = 25
DEFAULT_PREDICT_PERIOD = "1y"
DEFAULT_PREDICT_DAYS_AHEAD = 10
# compute_predict_algo_comparison used to run this fully sequentially — one
# ticker's model training + yfinance fetch, then the next. Fine for a
# handful of tickers, but a 15-16 position portfolio (or worse, an actively
# rate-limited Yahoo session where each ticker eats its full retry/backoff
# delay) turned into minutes of wall-clock time on /portfolio/insights.
# Bounded concurrency, same MAX_PARALLEL_FETCHES=4 used elsewhere after the
# rate-limit incident.
PREDICT_COMPARE_MAX_PARALLEL = 4
# Selectable horizons for comparing the Predict-page algorithm against the
# published momentum picks. Capped at 30 to match the published lookback
# window (comparing beyond that would forecast further out than the
# momentum return it's being set against).
PREDICT_COMPARE_HORIZONS = [1, 5, 10, 30]
# TR-4: outcomes are evaluated over the same window the picks were ranked
# on — symmetric with DEFAULT_LOOKBACK_DAYS, so "ranked by trailing 30d"
# is checked against "realized over the following 30 trading days."
DEFAULT_HORIZON_DAYS = 30
# TRK-2 (docs/stock-analysis-requirements.html): track-record metrics are
# grouped by horizon per the doc's section 4 ("grouped by horizon (10, 30,
# 60, 90 days)") — signal_outcomes' schema already supports this (its
# unique constraint includes horizon_days), so this is just which values
# the nightly evaluation job and the track-record endpoint loop over.
TRACK_RECORD_HORIZONS = [10, 30, 60, 90]
# TRK-3: confidence buckets, half-open [low, high) except the last.
CONFIDENCE_BUCKETS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
# TRK-6: growth-of-$10,000 starting value, same convention as
# portfolio_compare_service.REBASE_TO.
MODEL_PORTFOLIO_REBASE_TO = 10_000.0


@lru_cache(maxsize=1)
def get_model_version_hash() -> str:
    """
    The publication's model_version_hash (TR-2): the git commit the running
    code was built from. The published Signal Set is a pure, deterministic
    function of PIT price data and this codebase — no separately-trained
    model artifact to version — so the commit hash IS the model version.
    Cached for the process lifetime since it can't change without a redeploy.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning("Could not resolve git commit hash: %s", e)
        return "unknown"


def build_daily_signal_set(
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    Deterministic top-N momentum ranking — the same trailing-return sort
    already shown on /top-performers, reused here rather than reimplemented
    so the published record and the live leaderboard are provably the same
    rule. Pure function of PIT price data: no fundamentals, no training, no
    randomness — reconstructible at any date given the same price history.
    """
    col = f"Return {lookback_days}D %"
    df = get_stock_finder_table(universe_id)
    if df.empty or col not in df.columns:
        return []

    ranked = df.dropna(subset=[col]).sort_values(col, ascending=False).head(top_n)
    return [
        {
            "rank": i + 1,
            "ticker": str(row["Ticker"]),
            "trailing_return_pct": round(float(row[col]), 4),
        }
        for i, (_, row) in enumerate(ranked.iterrows())
    ]


def build_daily_signal_set_hybrid(
    pit_price_rows: list[dict],
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict]:
    """
    TR-2: the same ranking rule as build_daily_signal_set, but sources each
    ticker's trailing-return figure from the PIT store whenever the PIT
    store already has lookback_days + 1 days of price history for that
    ticker, falling back to today's live yfinance fetch only for tickers
    the PIT store can't cover yet (merge_pit_and_live_scores does the
    actual merge/rank — kept pure and separately tested).

    This exists so publication never has to stop and wait for the PIT
    store to accumulate lookback_days + 1 trading days of history before
    it can run at all — a hard PIT-only cutover today would publish zero
    rows and stay that way for weeks. Instead, live-fallback usage shrinks
    on its own as PIT history deepens, until every ticker qualifies and
    this becomes provably identical to a PIT-only computation — which is
    exactly what GET /api/v1/pit-prices/reconcile already measures.
    """
    col = f"Return {lookback_days}D %"
    df = get_stock_finder_table(universe_id)
    live_scored = []
    if not df.empty and col in df.columns:
        for _, row in df.dropna(subset=[col]).iterrows():
            live_scored.append({"ticker": str(row["Ticker"]), "trailing_return_pct": round(float(row[col]), 4)})

    pit_scored = score_tickers_from_pit(pit_price_rows, lookback_days)
    return merge_pit_and_live_scores(pit_scored, live_scored, top_n)


def rank_within_universe(
    tickers: list[str],
    universe_id: str = DEFAULT_UNIVERSE,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, dict]:
    """
    Each of the given tickers' rank (1 = best trailing return) within the
    full universe — not capped to a top_n like build_daily_signal_set,
    since a held portfolio position might rank anywhere, not just in the
    top 25. rank_tickers_against_universe does the actual ranking (pure,
    separately tested); this just supplies the live universe table.
    """
    col = f"Return {lookback_days}D %"
    df = get_stock_finder_table(universe_id)
    universe_rows = []
    if not df.empty and col in df.columns:
        for _, row in df.dropna(subset=[col]).iterrows():
            universe_rows.append({"ticker": str(row["Ticker"]), "trailing_return_pct": round(float(row[col]), 4)})
    return rank_tickers_against_universe(universe_rows, tickers)


def compute_predict_algo_comparison(
    tickers: list[str],
    period: str = DEFAULT_PREDICT_PERIOD,
    days_ahead: int = DEFAULT_PREDICT_DAYS_AHEAD,
    extra_horizons: list[int] | None = None,
) -> list[dict]:
    """
    For each ticker, runs the same trained-model algorithm used on /predict
    (predict_future_prices + generate_trading_signal) — a completely
    different, non-deterministic signal from the momentum ranking above —
    so a reader can see what that separate model currently says about
    today's published picks.

    Only valid against the *current* (latest) publication: this always
    reflects today's price data, so running it against an older published
    date would silently use data the model couldn't have had at that
    original date — there's no point-in-time store yet to prevent that
    honestly. Callers must not offer this for historical dates.

    extra_horizons additionally reports the same model's forecast at each
    given number of trading days out (e.g. [1, 5]), as
    predict_target_price_{n}d / predict_expected_return_pct_{n}d — read
    from the same future_df already computed for the days_ahead-out
    signal above, no extra model call. None (the default) keeps existing
    callers' row shape unchanged. Each n must be <= days_ahead.

    Tickers are fanned out across a small thread pool (PREDICT_COMPARE_MAX_PARALLEL)
    rather than processed one at a time — see that constant's comment.
    Callers key the result list by ticker, so return order doesn't matter.
    """

    def _row_for(ticker: str) -> dict:
        last_close = get_latest_price(ticker)
        future_df = predict_future_prices(ticker, period, days_ahead, False)
        if last_close is None or future_df is None or future_df.empty:
            row = {
                "ticker": ticker,
                "predict_signal": None,
                "predict_expected_return_pct": None,
                "predict_target_price": None,
            }
            for n in extra_horizons or []:
                row[f"predict_target_price_{n}d"] = None
                row[f"predict_expected_return_pct_{n}d"] = None
            return row

        sig = generate_trading_signal(last_close, future_df)
        row = {
            "ticker": ticker,
            "predict_signal": sig.get("signal"),
            "predict_expected_return_pct": sig.get("expected_return_pct"),
            "predict_target_price": sig.get("target_price"),
        }
        for n in extra_horizons or []:
            if len(future_df) >= n:
                price_n = float(future_df["Predicted"].iloc[n - 1])
                row[f"predict_target_price_{n}d"] = round(price_n, 2)
                row[f"predict_expected_return_pct_{n}d"] = round((price_n - last_close) / last_close * 100.0, 2)
            else:
                row[f"predict_target_price_{n}d"] = None
                row[f"predict_expected_return_pct_{n}d"] = None
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=PREDICT_COMPARE_MAX_PARALLEL) as executor:
        futures = [executor.submit(_row_for, ticker) for ticker in tickers]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def evaluate_signal_outcomes_for_date(
    target_date_rows: list[dict],
    target_date: datetime.date,
    universe_id: str,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> Optional[list[dict]]:
    """
    TR-4: the live, out-of-sample counterpart to backtest_momentum_ranking
    — same methodology (entry at the publication date's close, exit
    `horizon_days` trading days later, equal-weight-universe benchmark over
    the identical window), applied to one already-published date's real
    picks instead of a simulated walk-forward. Returns None if `horizon_days`
    trading days haven't actually elapsed since target_date yet — an
    outcome that isn't knowable yet is never guessed at or padded in.

    Uses _universe_tickers (not the raw STOCK_UNIVERSES dict) — same fix,
    same reason, as pit_price_service.py's capture_universe_closes and
    pit_fundamentals_service.py's capture_universe_fundamentals: "All" and
    "US - S&P 500" resolve their real ticker lists lazily (a live, cached
    Wikipedia fetch), so STOCK_UNIVERSES itself holds only empty
    placeholders for those two keys. Reading it directly here silently
    returned zero tickers for every "All"-universe evaluation call, which
    is why signal_outcomes had zero rows despite ~2 months of publication
    history — found live while verifying Stage B, not introduced by it.
    """
    tickers = list(_universe_tickers(universe_id))
    if not tickers:
        return None

    raw = yf.download(tickers, period="1y", auto_adjust=True, progress=False, group_by="ticker")
    closes: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            series = raw[t]["Close"].dropna() if len(tickers) > 1 else raw["Close"].dropna()
            if not series.empty:
                closes[t] = series
        except Exception:
            continue
    if not closes:
        return None

    common_index = None
    for s in closes.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    common_index = common_index.sort_values()
    if common_index.empty:
        return None

    target_ts = pd.Timestamp(target_date)
    on_or_after = common_index[common_index >= target_ts]
    if on_or_after.empty:
        return None
    entry_idx = common_index.get_loc(on_or_after[0])

    exit_idx = entry_idx + horizon_days
    if exit_idx >= len(common_index):
        return None  # not enough trading days have elapsed yet — not due

    entry_date = common_index[entry_idx]
    exit_date = common_index[exit_idx]

    price_matrix = pd.DataFrame({t: s.reindex(common_index) for t, s in closes.items()})
    entry_prices = price_matrix.loc[entry_date]
    exit_prices = price_matrix.loc[exit_date]

    universe_returns = (exit_prices / entry_prices - 1.0).dropna()
    if universe_returns.empty:
        return None
    benchmark_return_pct = float(universe_returns.mean() * 100)

    outcomes = []
    for row in target_date_rows:
        ticker = row["ticker"]
        entry_price = entry_prices.get(ticker)
        exit_price = exit_prices.get(ticker)
        if entry_price is None or exit_price is None or pd.isna(entry_price) or pd.isna(exit_price):
            continue
        realized_return_pct = (float(exit_price) / float(entry_price) - 1.0) * 100
        outcomes.append(
            {
                "ticker": ticker,
                "rank": row["rank"],
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "realized_return_pct": round(realized_return_pct, 4),
                "benchmark_return_pct": round(benchmark_return_pct, 4),
                "beat_benchmark": realized_return_pct > benchmark_return_pct,
            }
        )
    return outcomes if outcomes else None


def compute_outcome_metrics(outcome_rows: list[dict]) -> dict:
    """
    TR-4's standard metrics, computed from already-evaluated signal_outcomes
    rows (each carrying target_date, rank, realized_return_pct,
    beat_benchmark). Ranking-signal counterparts of the doc's generic
    "hit rate / mean & median error / information coefficient / decile
    spread" — this is a ranking, not a point prediction, so "error" isn't
    well-defined, but the other three translate directly:

    - hit_rate_pct: share of individual picks that beat the equal-weight
      universe benchmark over their evaluation window.
    - information_coefficient: mean, across evaluation dates, of the rank
      correlation between assigned rank and realized forward return
      (Spearman's rho via rank-then-Pearson, no extra dependency) —
      positive means "lower rank number (better momentum) really did
      predict higher forward return."
    - quintile_spread_pct: mean, across evaluation dates, of (average
      return of the best-ranked fifth minus the worst-ranked fifth).
      Reported as quintiles rather than deciles since the published
      universe is 25 names — deciles would only hold ~2-3 names each,
      too few to be a stable comparison.
    """
    if not outcome_rows:
        return {
            "num_evaluated_dates": 0,
            "num_evaluated_picks": 0,
            "hit_rate_pct": None,
            "avg_return_pct": None,
            "information_coefficient": None,
            "quintile_spread_pct": None,
        }

    df = pd.DataFrame(outcome_rows)
    hit_rate_pct = round(float(df["beat_benchmark"].mean()) * 100, 1)
    # TRK-2: mean realized return across every evaluated pick -- already
    # stored per row, zero new capture needed.
    avg_return_pct = round(float(df["realized_return_pct"].mean()), 2)

    ics = []
    spreads = []
    for _, group in df.groupby("target_date"):
        if len(group) >= 5:
            # Spearman = Pearson correlation of the ranks. Negated since a
            # *lower* rank number (rank 1 = best momentum) should correlate
            # with *higher* realized return — flipping sign makes positive
            # IC mean "the signal worked," the standard quant convention.
            rank_corr = group["rank"].rank().corr(group["realized_return_pct"].rank())
            if rank_corr is not None and not pd.isna(rank_corr):
                ics.append(-float(rank_corr))

            sorted_group = group.sort_values("rank")
            quintile_size = max(1, len(sorted_group) // 5)
            best = sorted_group.head(quintile_size)["realized_return_pct"].mean()
            worst = sorted_group.tail(quintile_size)["realized_return_pct"].mean()
            if pd.notna(best) and pd.notna(worst):
                spreads.append(float(best) - float(worst))

    return {
        "num_evaluated_dates": int(df["target_date"].nunique()),
        "num_evaluated_picks": int(len(df)),
        "hit_rate_pct": hit_rate_pct,
        "avg_return_pct": avg_return_pct,
        "information_coefficient": round(sum(ics) / len(ics), 4) if ics else None,
        "quintile_spread_pct": round(sum(spreads) / len(spreads), 2) if spreads else None,
    }


def compute_outcome_metrics_by_model_version(outcome_rows: list[dict]) -> dict[str, dict]:
    """TRK-2: compute_outcome_metrics, grouped by each row's
    model_version_hash (the caller joins this in from published_signals --
    this module has no DB access of its own). A model-version change
    starts a new record per the doc's own convention (section 4: "older
    versions stay visible"), so versions are reported separately here,
    never merged together."""
    by_version: dict[str, list[dict]] = {}
    for row in outcome_rows:
        by_version.setdefault(row.get("model_version_hash") or "unknown", []).append(row)
    return {version: compute_outcome_metrics(rows) for version, rows in by_version.items()}


def worst_misses(outcome_rows: list[dict], top_n: int = 10) -> list[dict]:
    """TRK-5: the largest losses among published picks, sorted ascending
    by realized_return_pct. published_signals has no Trim/Sell side
    (confirmed: no such concept exists anywhere in this publication
    pipeline) -- this can only ever show the Buy-side half of TRK-5's
    acceptance criterion ("largest losses on Buy signals and largest
    gains on Trim signals"); callers must say so explicitly rather than
    silently presenting this as the complete picture."""
    return sorted(outcome_rows, key=lambda r: r["realized_return_pct"])[:top_n]


def fetch_spy_close_series(period: str = "2y") -> pd.Series:
    """Live SPY close history via the shared, cached yfinance wrapper
    (services/yfinance_cache.py) -- not the raw yf.download this file uses
    for the full universe elsewhere, since SPY alone is small, requested
    on every /outcomes-style call, and worth deduping across users the
    same way portfolio_compare_service.py already does for its own SPY
    fetch."""
    hist = get_cached_history("SPY", period, auto_adjust=True)
    if hist.empty:
        return pd.Series(dtype=float)
    return hist["Close"].dropna()


def compute_spy_returns_for_dates(
    spy_close: pd.Series, target_dates: list[datetime.date], horizon_days: int
) -> dict[datetime.date, Optional[float]]:
    """TRK-2: SPY's own return over the IDENTICAL window each target_date's
    picks were evaluated over -- entry = first SPY trading day on/after
    target_date, exit = horizon_days trading days later. Mirrors
    evaluate_signal_outcomes_for_date's own entry/exit walk exactly, so
    "excess vs SPY" is a true apples-to-apples comparison, not just a
    same-period approximation. None for a date SPY's own history can't
    yet resolve (window not fully elapsed, or before SPY's history starts)."""
    index = spy_close.index.sort_values()
    result: dict[datetime.date, Optional[float]] = {}
    for target_date in target_dates:
        target_ts = pd.Timestamp(target_date)
        on_or_after = index[index >= target_ts]
        if on_or_after.empty:
            result[target_date] = None
            continue
        entry_idx = index.get_loc(on_or_after[0])
        exit_idx = entry_idx + horizon_days
        if exit_idx >= len(index):
            result[target_date] = None
            continue
        entry_price = float(spy_close.iloc[entry_idx])
        exit_price = float(spy_close.iloc[exit_idx])
        result[target_date] = round((exit_price / entry_price - 1.0) * 100, 4) if entry_price else None
    return result


def attach_excess_vs_spy(outcome_rows: list[dict], spy_return_by_date: dict) -> list[dict]:
    """TRK-2: adds excess_vs_spy_pct to each row (None when SPY's own
    window for that date couldn't be resolved) -- an ADDITIONAL
    comparison alongside the existing equal-weight-universe
    benchmark_return_pct, not a replacement for it."""
    result = []
    for row in outcome_rows:
        spy_return = spy_return_by_date.get(row["target_date"])
        excess = round(row["realized_return_pct"] - spy_return, 4) if spy_return is not None else None
        result.append({**row, "excess_vs_spy_pct": excess})
    return result


def compute_avg_excess_vs_spy(outcome_rows_with_excess: list[dict]) -> Optional[float]:
    values = [r["excess_vs_spy_pct"] for r in outcome_rows_with_excess if r.get("excess_vs_spy_pct") is not None]
    return round(sum(values) / len(values), 2) if values else None


def confidence_for_outcome(
    ticker_signal_history: list[tuple], target_date: datetime.date, lookback_days: int = 30
) -> dict:
    """TRK-3: a confidence PROXY for one outcome row, derived from
    pit_quant_signal's own flip-count stability as of target_date -- a
    DIFFERENT signal than the momentum ranking being calibrated here,
    reused only because it's the one signal-stability measure old enough
    to retroactively score the full existing signal_outcomes history
    against (Stage A's own two-score signal is too new to have any
    history yet). Reuses flip_count_from_signal_history (services/
    stock_score_service.py) and derive_confidence (services/
    portfolio_compare_service.py) rather than duplicating either formula.

    Never looks at a signal dated after target_date -- the same
    lookahead-safety boundary every PIT computation in this app respects;
    ticker_signal_history may contain later rows (the caller fetches one
    shared window per ticker across many outcome rows), so this function
    is what enforces the per-row cutoff.
    """
    window = [
        (d, s) for d, s in ticker_signal_history
        if d <= target_date and d > target_date - datetime.timedelta(days=lookback_days)
    ]
    stability = flip_count_from_signal_history(window)
    return derive_confidence(stability)


def compute_calibration(outcome_rows_with_confidence: list[dict]) -> list[dict]:
    """TRK-3: for each confidence bucket, the actual hit rate + sample
    size -- outcome_rows_with_confidence carries a `confidence_score`
    (0-100, or None when there wasn't enough signal history yet) per row
    from confidence_for_outcome above. Buckets with zero rows still
    appear (hit_rate_pct=None, sample_size=0) so the UI can show "not
    enough data yet" per bucket rather than silently omitting it."""
    buckets = []
    for low, high in CONFIDENCE_BUCKETS:
        in_bucket = [
            r for r in outcome_rows_with_confidence
            if r.get("confidence_score") is not None and low <= r["confidence_score"] < high
        ]
        label = f"{low}-{min(high, 100)}%"
        if not in_bucket:
            buckets.append({"bucket_label": label, "hit_rate_pct": None, "sample_size": 0})
            continue
        hit_rate_pct = round(sum(1 for r in in_bucket if r["beat_benchmark"]) / len(in_bucket) * 100, 1)
        buckets.append({"bucket_label": label, "hit_rate_pct": hit_rate_pct, "sample_size": len(in_bucket)})
    return buckets


def _select_non_overlapping_dates(outcome_rows: list[dict], horizon_days: int) -> list[datetime.date]:
    """Shared by build_model_portfolio_series and build_spy_comparison_series
    so both curves compound over the IDENTICAL set of dates and stay
    directly comparable point-for-point. Publication is daily but each
    pick's own evaluation window is horizon_days trading days long, so
    naively chaining every published date's cohort would double-count the
    same trading days many times over (each day's window overlaps
    horizon_days-1 other days' windows). Taking every horizon_days-th
    published date (by position in the sorted list of dates that
    actually have outcomes, which are already trading days) guarantees
    each selected cohort's window ends at or before the next one starts.
    """
    dates = sorted({row["target_date"] for row in outcome_rows})
    return dates[0::horizon_days] if dates else []


def build_model_portfolio_series(outcome_rows: list[dict], horizon_days: int) -> list[list]:
    """TRK-6: growth of $10,000 from equal-weight Buys, built from
    NON-OVERLAPPING evaluation windows only (see
    _select_non_overlapping_dates). Trades data density for an honest,
    non-overlapping compounding curve -- thin with only a few weeks of
    publication history, deepens as more full periods accumulate.
    Assumes zero trading costs; callers must label that.
    """
    by_date: dict[datetime.date, list[float]] = {}
    for row in outcome_rows:
        by_date.setdefault(row["target_date"], []).append(row["realized_return_pct"])
    selected = _select_non_overlapping_dates(outcome_rows, horizon_days)
    if not selected:
        return []

    value = MODEL_PORTFOLIO_REBASE_TO
    series = [[selected[0].isoformat(), round(value, 2)]]
    for d in selected:
        period_return_pct = sum(by_date[d]) / len(by_date[d])
        value *= 1 + period_return_pct / 100
        series.append([d.isoformat(), round(value, 2)])
    return series


def build_spy_comparison_series(outcome_rows: list[dict], spy_return_by_date: dict, horizon_days: int) -> list[list]:
    """TRK-6: SPY's own growth-of-$10,000 curve over the SAME
    non-overlapping dates build_model_portfolio_series selects (via
    _select_non_overlapping_dates), so the two series are directly
    comparable point-for-point on one chart. A period whose SPY return
    couldn't be resolved (compute_spy_returns_for_dates returned None)
    is skipped -- the curve holds its last value rather than guessing."""
    selected = _select_non_overlapping_dates(outcome_rows, horizon_days)
    if not selected:
        return []

    value = MODEL_PORTFOLIO_REBASE_TO
    series = [[selected[0].isoformat(), round(value, 2)]]
    for d in selected:
        spy_return_pct = spy_return_by_date.get(d)
        if spy_return_pct is not None:
            value *= 1 + spy_return_pct / 100
        series.append([d.isoformat(), round(value, 2)])
    return series
