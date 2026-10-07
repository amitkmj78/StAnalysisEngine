"""
AGT-30: the backtest leg of the live-trading validation gate.

Scope was narrowed twice before any of this was written, both narrowings
disclosed in run_agt30_stop_validation()'s own result rather than silently
assumed:

1. The regime-based exposure cap (services/market_regime_service.py) is
   excluded. That signal has already failed its own release gate three
   times, and its own most recent test -- isolating 2008 specifically --
   found extreme internals stress predicted a FURTHER -7.91% over the next
   21 days (n=24, p<0.0001): the opposite of what a Risk-On/Risk-Off
   exposure cap wants. Re-testing an already-disclosed-bad signal here
   would not be a real test, so this module holds exposure flat (not
   regime-capped) in both variants and tests the STOP-LOSS rule only
   (AGT-16/29's ATR trailing stop, services/agent/runner.py::
   _stop_trail_percent, reproduced here as a vectorized series).

2. A true survivorship-bias-free multi-stock universe back to 2008+ is not
   buildable from this app's data sources: yfinance cannot serve price
   history for S&P 500 members that have since been delisted (Lehman
   Brothers, Bear Stearns, Washington Mutual, ...), so there is no way to
   know what an agent scanning "today's tradable candidates" would
   actually have held in 2008 without survivorship bias creeping in on
   exactly the crisis the gate most needs to test. This module tests the
   stop-loss formula on SPY itself (continuous data since 1993, no
   delisting risk) instead.

A further, unavoidable substitution: the agent's real entry condition is a
Buy signal from the two-score system (services/stock_score_capture_
service.py), which has no historical record before that table started
(~Sept 2026) -- it cannot be replayed for 2008-2024. This module uses the
same price-vs-200-day-average trend filter services/agent/risk.py::
filter_candidates already applies as a Buy-signal-independent entry/exit
proxy, re-checked monthly like the agent's own rebalance.

What this DOES answer: does the agent's exact ATR trailing-stop formula
reduce max drawdown versus the same trend-following entry/exit without
it, over a window spanning 2008, 2020, and 2022, after per-trade costs?

What it does NOT answer: whether the agent's actual candidate-selection
logic would have performed well in 2008 (untestable -- no historical
signal data exists), or whether the regime exposure cap helps (already
disclosed as failed, see above). AGT-2's live-mode gate treats both of
those as open/failed, not as validated by this result.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from services.agent.config import CONFIG, AgentConfig, config_version
from services.backtest_engine import max_drawdown_pct

# Same per-trade cost convention AGT-32 already names for this agent's
# backtest assumptions (10bps), applied once on entry and once on exit.
DEFAULT_COST_BPS = 10.0

# The spec names these years; each window is widened a little past the
# calendar year so it actually captures the trough, not just the label.
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "2008_gfc": ("2007-10-01", "2009-03-31"),
    "2020_covid": ("2020-02-01", "2020-04-30"),
    "2022_bear": ("2022-01-01", "2022-12-31"),
}

MIN_YEARS_REQUIRED = 10.0


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Same simple-mean-of-true-range convention as services/agent/
    indicators.py::atr, carried forward as a rolling series instead of one
    latest value -- this needs "ATR as of every past day", not just today's."""
    return _true_range(high, low, close).rolling(window).mean()


def _stop_trail_pct(price: pd.Series, atr14: pd.Series, config: AgentConfig = CONFIG) -> pd.Series:
    """Vectorized form of services/agent/runner.py::_stop_trail_percent --
    kept numerically identical to the formula the agent actually submits
    to the broker, not a reinterpretation of it."""
    raw = config.stop_atr_multiple * atr14 / price * 100.0
    return raw.clip(lower=config.stop_min_pct, upper=config.stop_max_pct)


@dataclass
class SimResult:
    daily_returns_pct: list[float]
    dates: list[str]
    trade_count: int


def simulate_trend_following(
    df: pd.DataFrame,
    *,
    use_stop: bool,
    cost_bps: float = DEFAULT_COST_BPS,
    config: AgentConfig = CONFIG,
) -> SimResult:
    """In the market whenever price is above its trend_sma_days average
    (the agent's own trend filter, filter_candidates' sma200 check) at
    each month's first session, out otherwise -- a Buy-signal-independent
    proxy, see module docstring for why. When use_stop is True, the same
    ATR trailing-stop formula the agent submits to the broker is also
    checked every session (not just at rebalance), closing the position
    the day it fires rather than waiting for the next month-start check.

    df must have Close/High/Low columns and a DatetimeIndex, ascending,
    with no gaps the caller cares about preserved (NaNs in High/Low/Close
    are dropped up front). Returns daily %, 0.0 on days spent in cash.
    """
    df = df.dropna(subset=["Close", "High", "Low"]).copy()
    sma = df["Close"].rolling(config.trend_sma_days).mean()
    atr14 = _rolling_atr(df["High"], df["Low"], df["Close"], 14)
    trail_pct = _stop_trail_pct(df["Close"], atr14, config)

    periods = df.index.to_series().dt.to_period("M")
    is_month_start = (periods != periods.shift(1)).to_numpy()

    closes = df["Close"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    sma_vals = sma.to_numpy()
    trail_vals = trail_pct.to_numpy()

    in_position = False
    peak_since_entry: Optional[float] = None
    returns_pct: list[float] = []
    trade_count = 0

    for i in range(len(df)):
        price = closes[i]
        prev_price = closes[i - 1] if i > 0 else price
        day_return_pct = (price / prev_price - 1.0) * 100.0 if in_position else 0.0
        exited_today = False
        entered_today = False

        # 1. Stop check first, against today's LOW (a real trailing-stop
        #    broker order fires intraday the moment price crosses it, not
        #    only once the bar closes) -- realized at the stop price
        #    itself, not the close, same as a real stop fill.
        if use_stop and in_position and peak_since_entry and not math.isnan(trail_vals[i]):
            stop_price = peak_since_entry * (1.0 - trail_vals[i] / 100.0)
            if lows[i] <= stop_price:
                day_return_pct = (stop_price / prev_price - 1.0) * 100.0 - cost_bps / 100.0
                in_position = False
                peak_since_entry = None
                trade_count += 1
                exited_today = True

        # 2. Monthly rebalance (trend filter) -- skipped if the stop
        #    already closed the position today.
        if not exited_today and i > 0 and is_month_start[i] and not math.isnan(sma_vals[i]):
            want_in = price > sma_vals[i]
            if want_in and not in_position:
                in_position = True
                peak_since_entry = max(price, highs[i])
                day_return_pct -= cost_bps / 100.0
                trade_count += 1
                entered_today = True
            elif not want_in and in_position:
                in_position = False
                peak_since_entry = None
                day_return_pct -= cost_bps / 100.0
                trade_count += 1

        # 3. Track the running peak (by High, since a trailing stop trails
        #    the highest price actually reached, not just closes) for
        #    whatever position is open at day's end.
        if in_position and not entered_today:
            peak_since_entry = highs[i] if peak_since_entry is None else max(peak_since_entry, highs[i])

        returns_pct.append(day_return_pct)

    return SimResult(
        daily_returns_pct=returns_pct,
        dates=[d.strftime("%Y-%m-%d") for d in df.index],
        trade_count=trade_count,
    )


def _window_returns(returns: list[float], dates: list[str], start: str, end: str) -> list[float]:
    return [r for r, d in zip(returns, dates) if start <= d <= end]


def run_agt30_stop_validation(
    df: pd.DataFrame, cost_bps: float = DEFAULT_COST_BPS, config: AgentConfig = CONFIG
) -> dict:
    """The full (narrowed, disclosed) AGT-30 backtest report. df is one
    instrument's OHLC history, ascending by date -- callers pass SPY. See
    module docstring for exactly what this does and does not test."""
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    years = (df.index[-1] - df.index[0]).days / 365.25

    with_stop = simulate_trend_following(df, use_stop=True, cost_bps=cost_bps, config=config)
    without_stop = simulate_trend_following(df, use_stop=False, cost_bps=cost_bps, config=config)
    benchmark_returns = (df["Close"].pct_change().fillna(0.0) * 100.0).tolist()
    dates = with_stop.dates

    crisis_windows = {}
    for name, (start, end) in CRISIS_WINDOWS.items():
        crisis_windows[name] = {
            "with_stop_max_drawdown_pct": max_drawdown_pct(_window_returns(with_stop.daily_returns_pct, dates, start, end)),
            "without_stop_max_drawdown_pct": max_drawdown_pct(_window_returns(without_stop.daily_returns_pct, dates, start, end)),
            "spy_max_drawdown_pct": max_drawdown_pct(_window_returns(benchmark_returns, dates, start, end)),
        }

    full_with = max_drawdown_pct(with_stop.daily_returns_pct)
    full_without = max_drawdown_pct(without_stop.daily_returns_pct)
    full_spy = max_drawdown_pct(benchmark_returns)

    # max_drawdown_pct returns negative percentages (or 0.0) -- smaller
    # drawdown means a value closer to zero, i.e. numerically greater.
    # bool(...) here because numpy comparisons return numpy.bool_, which
    # json.dumps (and this function's own callers) can't serialize.
    beats_spy = bool(full_with is not None and full_spy is not None and full_with > full_spy)
    beats_no_stop_variant = bool(full_with is not None and full_without is not None and full_with > full_without)
    covers_required_years = bool(years >= MIN_YEARS_REQUIRED)
    covers_crisis_windows = bool(all(any(start <= d <= end for d in dates) for start, end in CRISIS_WINDOWS.values()))

    passed = bool(beats_spy and beats_no_stop_variant and covers_required_years and covers_crisis_windows)

    return {
        "scope": "stop_rule_only_on_spy",
        # Ties a stored result to the exact limits it was run under --
        # backtest_gate_leg() below treats a stored record from a
        # different config_version as stale, not as a current pass.
        "agent_config_version": config_version(config),
        "years_covered": round(years, 1),
        "covers_required_years": covers_required_years,
        "covers_crisis_windows": covers_crisis_windows,
        "cost_bps_per_trade": cost_bps,
        "full_period": {
            "with_stop_max_drawdown_pct": full_with,
            "without_stop_max_drawdown_pct": full_without,
            "spy_max_drawdown_pct": full_spy,
            "beats_spy": beats_spy,
            "beats_no_stop_variant": beats_no_stop_variant,
        },
        "crisis_windows": crisis_windows,
        "with_stop_trade_count": with_stop.trade_count,
        "without_stop_trade_count": without_stop.trade_count,
        "passed": passed,
        "excluded_from_this_test": [
            "Regime-based exposure cap: already failed its own validation gate three times, including "
            "a 2008 test showing it would have made things worse, not better. See services/"
            "market_regime_service.py's REGIME_METHODOLOGY. Re-testing it here would not be a real test.",
            "The agent's actual Buy/Hold/Trim signal: no historical record exists before the stock_scores "
            "table started (~Sept 2026), so it cannot be replayed for 2008-2024. A 200-day trend filter "
            "(the same one filter_candidates already applies) stands in for it here.",
            "A multi-stock universe: yfinance cannot serve price history for S&P 500 members that have "
            "since been delisted, so this tests the stop-loss formula on SPY itself (continuous data "
            "since 1993) rather than a survivorship-bias-free basket.",
        ],
    }


@dataclass
class LiveTradingGateStatus:
    """AGT-2's three gates, checked and reported individually rather than
    collapsed into one unconditional block -- each with its own plain
    reason, so a future re-check shows exactly what's still missing."""
    allow_live_trading_flag_set: bool
    backtest_validation_passed: Optional[bool]
    paper_trading_days: int
    paper_trading_meets_bar: Optional[bool]
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(
            self.allow_live_trading_flag_set
            and self.backtest_validation_passed
            and self.paper_trading_meets_bar
        )


# --- AGT-2's gate checks (I/O; everything above this line is pure) ---

ALLOW_LIVE_TRADING_ENV_VAR = "ALLOW_LIVE_TRADING"
AGT30_BACKTEST_VALIDATION_SETTING_KEY = "agt30_backtest_validation_json"
# The spec's "3 months" read literally as calendar days, not trading days.
PAPER_TRADING_MIN_DAYS = 90


def allow_live_trading_flag_set() -> bool:
    """The server-side flag AGT-2 names explicitly. Read fresh every call
    (not cached at import time) so flipping it in the environment and
    restarting the process is all it takes to change this -- no code
    change, but also nothing that can be toggled from inside the app."""
    return os.environ.get(ALLOW_LIVE_TRADING_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


async def backtest_gate_leg(config: AgentConfig = CONFIG) -> dict:
    """Reads back whatever run_agt30_stop_validation() result was last
    stored by POST /api/v1/admin/trading-agent/validate-agt30. A stored
    pass from before the agent's own config changed (e.g. a different
    stop_max_pct) is treated as stale, not as a current pass -- the
    config_version the record was run under must match CONFIG's own."""
    from web.backend.app_settings import get_setting_str

    raw = await get_setting_str(AGT30_BACKTEST_VALIDATION_SETTING_KEY, default="")
    if not raw:
        return {"passed": None, "reason": "No backtest validation has been run yet.", "record": None}
    try:
        stored = json.loads(raw)
    except ValueError:
        return {"passed": None, "reason": "Stored validation record is unreadable.", "record": None}

    if stored.get("agent_config_version") != config_version(config):
        return {
            "passed": None,
            "reason": "Stored validation was run under a different agent config version; re-run it.",
            "record": stored,
        }
    passed = bool(stored.get("passed"))
    return {
        "passed": passed,
        "reason": None if passed else "The stored backtest did not pass (see its own report for why).",
        "record": stored,
    }


async def paper_trading_gate_leg(user_id: str, config: AgentConfig = CONFIG) -> dict:
    """AGT-30's second leg: at least PAPER_TRADING_MIN_DAYS of Paper-mode
    runs for this user, with drawdown never breaching the configured
    circuit-breaker limit. Reads agent_runs' own per-run equity reading
    directly -- the same table AGT-27's journal already reports from --
    rather than requiring a separate equity-history table."""
    from web.backend.db import service_conn

    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT equity, created_at FROM agent_runs "
            "WHERE user_id = $1::uuid AND mode = 'paper' AND equity IS NOT NULL "
            "ORDER BY created_at",
            user_id,
        )
    if not rows:
        return {"days": 0, "max_drawdown_pct": None, "meets_bar": False, "reason": "No paper-mode runs on record yet."}

    days = (rows[-1]["created_at"].date() - rows[0]["created_at"].date()).days
    peak = float(rows[0]["equity"])
    max_dd_pct = 0.0
    for r in rows:
        equity = float(r["equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_dd_pct = min(max_dd_pct, (equity / peak - 1.0) * 100.0)

    enough_days = days >= PAPER_TRADING_MIN_DAYS
    within_limit = max_dd_pct >= -config.drawdown_breaker_pct
    if not enough_days:
        reason = f"{days} day(s) of paper-mode history on record (needs {PAPER_TRADING_MIN_DAYS}+)."
    elif not within_limit:
        reason = f"Drawdown reached {max_dd_pct:.1f}%, beyond the configured {config.drawdown_breaker_pct:g}% limit."
    else:
        reason = "Meets the bar."
    return {
        "days": days,
        "max_drawdown_pct": round(max_dd_pct, 2),
        "meets_bar": bool(enough_days and within_limit),
        "reason": reason,
    }


async def live_trading_gate_status(user_id: str, config: AgentConfig = CONFIG) -> LiveTradingGateStatus:
    """AGT-2's three gates, checked individually and together. Live stays
    blocked (see services/agent/runner.py's own unconditional block, which
    this does not touch) unless all three actually pass -- today, that
    means every call returns passed=False, since no paper-mode history
    exists yet regardless of the other two legs."""
    allow_flag = allow_live_trading_flag_set()
    backtest = await backtest_gate_leg(config)
    paper = await paper_trading_gate_leg(user_id, config)

    reasons: list[str] = []
    if not allow_flag:
        reasons.append(f"Server-side {ALLOW_LIVE_TRADING_ENV_VAR} is not set.")
    if not backtest["passed"]:
        reasons.append(f"Backtest validation (AGT-30): {backtest['reason']}")
    if not paper["meets_bar"]:
        reasons.append(f"Paper trading (AGT-30): {paper['reason']}")

    return LiveTradingGateStatus(
        allow_live_trading_flag_set=allow_flag,
        backtest_validation_passed=backtest["passed"],
        paper_trading_days=paper["days"],
        paper_trading_meets_bar=paper["meets_bar"],
        reasons=reasons,
    )
