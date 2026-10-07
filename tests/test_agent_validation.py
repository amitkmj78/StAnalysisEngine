import math

import numpy as np
import pandas as pd

from services.agent.config import CONFIG
from services.agent.validation import (
    CRISIS_WINDOWS,
    DEFAULT_COST_BPS,
    MIN_YEARS_REQUIRED,
    _rolling_atr,
    _stop_trail_pct,
    run_agt30_stop_validation,
    simulate_trend_following,
)
from services.backtest_engine import max_drawdown_pct


def _trending_then_crashing_ohlc(
    months: int = 30, crash_month: int = 20, crash_days: int = 15, daily_drop_pct: float = 3.0
) -> pd.DataFrame:
    """A steady uptrend (so the trend filter is long for most of the
    window) with a genuine multi-day decline partway through -- each day
    losing daily_drop_pct, well past what the ATR trailing-stop tolerates
    within the first few days -- then flat at the new, lower level. This
    is the case a trailing stop actually earns its keep on: exiting a few
    days into the decline avoids the REST of it, which a stop-free variant
    (only re-checked at the next month's rebalance) rides out in full."""
    dates = pd.bdate_range("2015-01-02", periods=months * 21, freq="B")
    base = 100.0 * (1.0 + 0.0006) ** np.arange(len(dates))  # gentle steady uptrend
    noise = 1.0 + 0.002 * np.sin(np.arange(len(dates)) / 5.0)  # tiny daily wiggle, keeps High>Low>0
    close = base * noise

    crash_start = crash_month * 21
    crash_end = crash_start + crash_days
    decline_factor = (1.0 - daily_drop_pct / 100.0) ** np.arange(1, crash_days + 1)
    close[crash_start:crash_end] = close[crash_start] * decline_factor
    close[crash_end:] = close[crash_end - 1]  # flat at the new, lower level afterward

    close = pd.Series(close, index=dates)
    high = close * 1.01
    low = close * 0.99
    open_ = close.copy()  # no gap by default; test_agt32... below overrides this where it matters
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def test_rolling_atr_matches_scalar_atr_at_the_tail():
    from services.agent.indicators import atr as scalar_atr

    df = _trending_then_crashing_ohlc()
    rolling = _rolling_atr(df["High"], df["Low"], df["Close"], 14)
    scalar = scalar_atr(df["High"], df["Low"], df["Close"], 14)
    assert math.isclose(rolling.iloc[-1], scalar, rel_tol=1e-9)


def test_stop_trail_pct_is_clamped_to_configured_min_and_max():
    price = pd.Series([100.0, 100.0])
    # ATR of 0 -> raw trail of 0%, clamped up to stop_min_pct.
    low_atr = pd.Series([0.0, 0.0])
    assert (_stop_trail_pct(price, low_atr) == CONFIG.stop_min_pct).all()
    # Huge ATR -> raw trail far above 100%, clamped down to stop_max_pct.
    huge_atr = pd.Series([50.0, 50.0])
    assert (_stop_trail_pct(price, huge_atr) == CONFIG.stop_max_pct).all()


def test_stop_variant_exits_the_decline_sooner_than_the_no_stop_variant():
    df = _trending_then_crashing_ohlc()
    with_stop = simulate_trend_following(df, use_stop=True)
    without_stop = simulate_trend_following(df, use_stop=False)

    crash_start = 20 * 21
    # The WITH-stop variant should close out partway through the 15-day
    # decline (once cumulative loss from the peak crosses its trail%),
    # well before the decline itself finishes -- i.e. somewhere in this
    # variant's own return series, a day inside the crash window has a
    # return of exactly 0.0 (flat, sitting in cash) while the decline is
    # still ongoing for the WITHOUT-stop variant.
    with_stop_went_flat_during_decline = any(
        r == 0.0 for r in with_stop.daily_returns_pct[crash_start : crash_start + 15]
    )
    without_stop_still_falling_throughout = all(
        r < 0.0 for r in without_stop.daily_returns_pct[crash_start : crash_start + 15]
    )
    assert with_stop_went_flat_during_decline
    assert without_stop_still_falling_throughout


def test_stop_reduces_max_drawdown_on_a_sharp_crash():
    df = _trending_then_crashing_ohlc()
    with_stop = simulate_trend_following(df, use_stop=True)
    without_stop = simulate_trend_following(df, use_stop=False)

    dd_with = max_drawdown_pct(with_stop.daily_returns_pct)
    dd_without = max_drawdown_pct(without_stop.daily_returns_pct)
    # Less negative == smaller drawdown.
    assert dd_with > dd_without


def test_agt32_stop_fills_at_the_gapped_down_open_not_the_theoretical_stop_price():
    """AGT-32: a stop exit must be modeled at the next available price,
    not an idealized fill exactly at the stop level. Here the triggering
    day's entire bar gaps down far below any plausible stop level -- the
    realized fill must be that (worse) open, same min(open, level)
    convention services/strategy_engine.py already uses for its own
    protective exits."""
    df = _trending_then_crashing_ohlc()
    trigger_day = 20 * 21  # well into the established uptrend, a position is already held here
    prev_close = df["Close"].iloc[trigger_day - 1]

    # Replace the whole bar with a severe, internally-consistent overnight
    # gap: Low is the day's minimum, Open/Close/High sit above it, and all
    # four are far below any stop level the earlier uptrend could produce
    # (stop_max_pct caps the trail at 15%, so even the loosest stop sits
    # well above half of the prior close).
    gapped = df.copy()
    cols = ["Open", "High", "Low", "Close"]
    gapped.iloc[trigger_day, [gapped.columns.get_loc(c) for c in cols]] = [
        prev_close * 0.48, prev_close * 0.52, prev_close * 0.45, prev_close * 0.50,
    ]

    with_stop = simulate_trend_following(gapped, use_stop=True)

    gapped_open = gapped["Open"].iloc[trigger_day]
    realized_return_pct = with_stop.daily_returns_pct[trigger_day]
    worst_case_return_pct = (gapped_open / prev_close - 1.0) * 100.0 - DEFAULT_COST_BPS / 100.0
    assert math.isclose(realized_return_pct, worst_case_return_pct, rel_tol=1e-6)


def test_no_stop_variant_ignores_the_trail_percent_entirely():
    df = _trending_then_crashing_ohlc()
    without_stop = simulate_trend_following(df, use_stop=False)
    # Only the monthly rebalance should ever close a position in this
    # variant -- trade_count can't exceed roughly 2 per entry/exit pair
    # across the ~30 months in the fixture.
    assert without_stop.trade_count <= 6


def test_run_agt30_stop_validation_discloses_its_narrowed_scope():
    df = _trending_then_crashing_ohlc(months=140, crash_month=20)  # ~11 years, comfortably over the bar
    df.index = pd.bdate_range("2007-01-02", periods=len(df), freq="B")
    report = run_agt30_stop_validation(df)

    assert report["scope"] == "stop_rule_only_on_spy"
    assert report["years_covered"] >= MIN_YEARS_REQUIRED
    assert report["covers_required_years"] is True
    assert len(report["excluded_from_this_test"]) == 3
    assert set(report["crisis_windows"]) == set(CRISIS_WINDOWS)
    assert "beats_spy" in report["full_period"]
    assert "beats_no_stop_variant" in report["full_period"]
    assert isinstance(report["passed"], bool)


def test_run_agt30_stop_validation_fails_the_years_check_on_a_short_window():
    df = _trending_then_crashing_ohlc(months=24, crash_month=12)  # ~2 years, under the 10yr bar
    report = run_agt30_stop_validation(df)
    assert report["covers_required_years"] is False
    assert report["passed"] is False
