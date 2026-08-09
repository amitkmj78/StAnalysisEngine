import datetime
import logging
import subprocess
from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf

from .data_service import get_latest_price
from .pit_signal_service import merge_pit_and_live_scores, score_tickers_from_pit
from .prediction_service import generate_trading_signal, predict_future_prices
from .stock_finder_service import STOCK_UNIVERSES, get_stock_finder_table

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = "All"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TOP_N = 25
DEFAULT_PREDICT_PERIOD = "1y"
DEFAULT_PREDICT_DAYS_AHEAD = 10
# Selectable horizons for comparing the Predict-page algorithm against the
# published momentum picks. Capped at 30 to match the published lookback
# window (comparing beyond that would forecast further out than the
# momentum return it's being set against).
PREDICT_COMPARE_HORIZONS = [1, 5, 10, 30]
# TR-4: outcomes are evaluated over the same window the picks were ranked
# on — symmetric with DEFAULT_LOOKBACK_DAYS, so "ranked by trailing 30d"
# is checked against "realized over the following 30 trading days."
DEFAULT_HORIZON_DAYS = 30


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


def compute_predict_algo_comparison(
    tickers: list[str],
    period: str = DEFAULT_PREDICT_PERIOD,
    days_ahead: int = DEFAULT_PREDICT_DAYS_AHEAD,
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
    """
    rows = []
    for ticker in tickers:
        last_close = get_latest_price(ticker)
        future_df = predict_future_prices(ticker, period, days_ahead, False)
        if last_close is None or future_df is None or future_df.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "predict_signal": None,
                    "predict_expected_return_pct": None,
                    "predict_target_price": None,
                }
            )
            continue

        sig = generate_trading_signal(last_close, future_df)
        rows.append(
            {
                "ticker": ticker,
                "predict_signal": sig.get("signal"),
                "predict_expected_return_pct": sig.get("expected_return_pct"),
                "predict_target_price": sig.get("target_price"),
            }
        )
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
    """
    tickers = list(STOCK_UNIVERSES.get(universe_id, []))
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
            "information_coefficient": None,
            "quintile_spread_pct": None,
        }

    df = pd.DataFrame(outcome_rows)
    hit_rate_pct = round(float(df["beat_benchmark"].mean()) * 100, 1)

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
        "information_coefficient": round(sum(ics) / len(ics), 4) if ics else None,
        "quintile_spread_pct": round(sum(spreads) / len(spreads), 2) if spreads else None,
    }
