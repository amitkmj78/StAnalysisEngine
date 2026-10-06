"""Strategy Builder v2 engine: rules, protective exits, cooldown, same-basket benchmark, checks.

Long only, daily bars, equal-weight sleeves (one per ticker, 1/N of equity each).

Timing, stated so the numbers can be checked:
  - Entry and indicator-exit rules are read at each day's CLOSE and fill at the NEXT open.
  - Protective exits are checked on each bar after the entry bar, before that bar's close:
    a stop fills at the stop price, or at the open when the bar gaps through it. If a stop and a
    take-profit are both reached in one bar, the stop is taken (conservative).
  - A time stop fills at the next open after the Nth session held.
  - Every fill pays `cost_bps + slippage_bps` on the sleeve.
Re-entry: after an exit the same ticker waits `cooldown` sessions, then needs a fresh entry event.
Benchmark: equal-weight buy-and-hold of the same tickers, rebalanced monthly, with the same costs.
"""

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from services.backtest_engine import cumulative_pct, max_drawdown_pct, sharpe
from services.chart_indicators import atr, rsi, sma
from services.strategy_robustness import deflated_sharpe, sensitivity, walk_forward

PERIODS_PER_YEAR = 252
IS_FRACTION = 0.7
DEFAULT_COST_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COOLDOWN = 5
MAX_TICKERS = 20
MANY_VARIANTS = 10
MIN_ROUND_TRIPS = 30
CHURN_WINDOW = 2
CHURN_WARN_PCT = 20.0
OOS_STRONGER_GAP = 0.8

NUMERIC_FIELDS = {
    "rsi_14": "RSI (14)",
    "close_vs_sma_50_pct": "Price vs 50-day average (%)",
    "close_vs_sma_200_pct": "Price vs 200-day average (%)",
    "sma_20_vs_50_pct": "20-day vs 50-day average (%)",
    "dist_52w_high_pct": "Distance from 52-week high (%)",
    "volume_vs_20d_pct": "Volume vs 20-day average (%)",
    "atr_14_pct": "ATR (14) as % of price",
    "sessions_since_earnings": "Sessions since the last earnings report",
}
REGIME_FIELD = "regime"
REGIME_LABELS = ("Risk-On", "Constructive", "Neutral", "Cautious", "Risk-Off")
NUMERIC_OPS = {">", "<", ">=", "<=", "crosses_above", "crosses_below"}
CATEGORY_OPS = {"is", "is_not"}


@dataclass(frozen=True)
class Rule:
    field: str
    op: str
    value: object

    @staticmethod
    def parse(raw: dict) -> "Rule":
        field_, op, value = raw.get("field"), raw.get("op"), raw.get("value")
        if field_ == REGIME_FIELD:
            if op not in CATEGORY_OPS or value not in REGIME_LABELS:
                raise ValueError("regime rules compare to one of the regime labels with 'is' or 'is_not'")
        elif field_ in NUMERIC_FIELDS:
            if op not in NUMERIC_OPS:
                raise ValueError(f"{field_} supports >, <, >=, <=, crosses_above, crosses_below")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{field_} needs a number")
        else:
            raise ValueError(f"unknown field: {field_}")
        return Rule(field_, op, value)


@dataclass(frozen=True)
class ProtectiveExits:
    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    atr_stop_k: Optional[float] = None
    time_stop_sessions: Optional[int] = None
    take_profit_pct: Optional[float] = None

    @staticmethod
    def parse(raw: Optional[dict]) -> "ProtectiveExits":
        raw = raw or {}

        def num(key: str, lo: float, hi: float) -> Optional[float]:
            v = raw.get(key)
            if v is None or v == "":
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{key} needs a number")
            if not lo <= v <= hi:
                raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
            return v

        time_stop = num("time_stop_sessions", 1, 2000)
        return ProtectiveExits(
            stop_loss_pct=num("stop_loss_pct", 0.1, 90),
            trailing_stop_pct=num("trailing_stop_pct", 0.1, 90),
            atr_stop_k=num("atr_stop_k", 0.1, 20),
            time_stop_sessions=int(time_stop) if time_stop is not None else None,
            take_profit_pct=num("take_profit_pct", 0.1, 1000),
        )

    def is_protective(self) -> bool:
        return any(v is not None for v in (self.stop_loss_pct, self.trailing_stop_pct, self.atr_stop_k, self.time_stop_sessions))


def _naive_dates(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def sessions_since_reports(index, report_dates) -> np.ndarray:
    """STB-1: sessions since the most recent earnings report, point-in-time. A report counts from the session AFTER its
    own date (the release may come after the close), so a day never sees a report that came later. NaN before the first."""
    out = np.full(len(index), np.nan)
    if len(report_dates) == 0:
        return out
    sessions = _naive_dates(index)
    for report in sorted(_naive_dates(report_dates)):
        start = sessions.searchsorted(report, side="right")
        if start < len(sessions):
            out[start:] = np.arange(len(sessions) - start)
    return out


def feature_frame(prices: pd.DataFrame, regime_by_date: Optional[dict[str, str]] = None,
                  earnings_reports: Optional[list] = None, member_flags: Optional[pd.Series] = None) -> pd.DataFrame:
    """prices: Open, High, Low, Close and optionally Volume (DatetimeIndex). Every value uses only
    data up to and including that session's close."""
    close = prices["Close"].astype(float)
    high = prices["High"].astype(float) if "High" in prices else close
    low = prices["Low"].astype(float) if "Low" in prices else close
    volume = prices["Volume"].astype(float) if "Volume" in prices else pd.Series(np.nan, index=prices.index)
    sma50 = sma(close, 50)
    out = pd.DataFrame(index=prices.index)
    out["open"] = prices["Open"].astype(float)
    out["high"] = high
    out["low"] = low
    out["close"] = close
    out["rsi_14"] = rsi(close, 14)
    out["close_vs_sma_50_pct"] = (close / sma50 - 1) * 100
    out["close_vs_sma_200_pct"] = (close / sma(close, 200) - 1) * 100
    out["sma_20_vs_50_pct"] = (sma(close, 20) / sma50 - 1) * 100
    out["dist_52w_high_pct"] = (close / close.rolling(252, min_periods=60).max() - 1) * 100
    out["volume_vs_20d_pct"] = (volume / volume.rolling(20).mean() - 1) * 100
    out["atr_14"] = atr(high, low, close, 14)
    out["atr_14_pct"] = out["atr_14"] / close * 100
    out["sessions_since_earnings"] = sessions_since_reports(prices.index, earnings_reports or [])
    # SCAN-2: a stock can only be bought while it was an index member; exits still apply to a position already held.
    out["is_member"] = member_flags.reindex(prices.index).fillna(False).astype(bool).to_numpy() if member_flags is not None else True
    if regime_by_date is not None:
        keys = [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in prices.index]
        out["regime"] = [regime_by_date.get(k) for k in keys]
    else:
        out["regime"] = None
    return out


def rule_mask(frame: pd.DataFrame, rule: Rule) -> pd.Series:
    if rule.field == REGIME_FIELD:
        is_match = frame["regime"] == rule.value
        return is_match if rule.op == "is" else ~is_match & frame["regime"].notna()
    series = frame[rule.field]
    if rule.op == ">":
        return series > rule.value
    if rule.op == "<":
        return series < rule.value
    if rule.op == ">=":
        return series >= rule.value
    if rule.op == "<=":
        return series <= rule.value
    prev = series.shift(1)
    if rule.op == "crosses_above":
        return (prev <= rule.value) & (series > rule.value)
    return (prev >= rule.value) & (series < rule.value)


def _all(frame: pd.DataFrame, rules: list[Rule]) -> np.ndarray:
    mask = pd.Series(True, index=frame.index)
    for r in rules:
        mask &= rule_mask(frame, r).fillna(False).astype(bool)
    return mask.to_numpy(bool)


def _any(frame: pd.DataFrame, rules: list[Rule]) -> np.ndarray:
    mask = pd.Series(False, index=frame.index)
    for r in rules:
        mask |= rule_mask(frame, r).fillna(False).astype(bool)
    return mask.to_numpy(bool)


def state_warnings(entry: list[Rule], exit_: list[Rule]) -> list[str]:
    """SB-R5: an exit threshold that the entry state already satisfies would re-enter at once."""
    out = []
    for e in entry:
        if e.op not in (">", ">=", "<", "<="):
            continue
        for x in exit_:
            if x.field != e.field or x.op != e.op:
                continue
            if (e.op in (">", ">=") and x.value >= e.value) or (e.op in ("<", "<=") and x.value <= e.value):
                out.append(
                    f"An exit at {x.field} {x.op} {x.value:g} also meets the entry condition "
                    f"({e.field} {e.op} {e.value:g}). Switch the entry to crosses above or below to stop re-buying at once."
                )
    return out


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str
    holding_days: int
    return_pct: float


@dataclass
class TickerRun:
    returns: np.ndarray        # sleeve daily return, fraction (costs included)
    costs: np.ndarray          # cost on the sleeve on each day, fraction
    held: np.ndarray           # 1 when held through the day
    trades: list = field(default_factory=list)
    exit_events: int = 0
    churn_events: int = 0


def _run_ticker(frame: pd.DataFrame, entry: list[Rule], exit_: list[Rule],
                exits: ProtectiveExits, cooldown: int, per_side: float, ticker: str) -> TickerRun:
    n = len(frame)
    idx = frame.index
    close = frame["close"].to_numpy(float)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    atr_prev = frame["atr_14"].shift(1).to_numpy(float)
    enter_sig = _all(frame, entry) if entry else np.zeros(n, bool)
    if "is_member" in frame.columns and not frame["is_member"].all():
        enter_sig = enter_sig & frame["is_member"].to_numpy(bool)
    exit_sig = _any(frame, exit_) if exit_ else np.zeros(n, bool)

    ret = np.zeros(n)
    cost = np.zeros(n)
    held = np.zeros(n, dtype=int)
    trades: list[Trade] = []
    exit_events = churn_events = 0
    last_exit = None

    in_pos = False
    entry_i = -1
    entry_px = 0.0
    peak = 0.0
    atr_entry = np.nan
    pending_entry = False
    pending_exit: Optional[str] = None
    cooldown_until = 0

    def close_trade(i: int, fill: float, reason: str) -> None:
        trades.append(Trade(
            ticker=ticker,
            entry_date=idx[entry_i],
            entry_price=round(entry_px, 4),
            exit_date=idx[i],
            exit_price=round(fill, 4),
            exit_reason=reason,
            holding_days=i - entry_i,
            return_pct=round((fill / entry_px - 1) * 100, 4),
        ))

    for i in range(n):
        if i > 0 and pending_exit is not None and in_pos:
            # indicator or time exit from the previous close: fill at this open
            fill = opens[i]
            ret[i] = fill / close[i - 1] - 1 - per_side
            cost[i] = per_side
            close_trade(i, fill, pending_exit)
            in_pos, pending_exit = False, None
            last_exit = i
            exit_events += 1
            cooldown_until = i + cooldown
            continue

        if i > 0 and pending_entry and not in_pos:
            # entry signal from the previous close: fill at this open
            pending_entry = False
            in_pos = True
            entry_i = i
            entry_px = opens[i]
            peak = highs[i]
            atr_entry = atr_prev[i]
            if last_exit is not None and i - last_exit <= CHURN_WINDOW:
                churn_events += 1
            ret[i] = close[i] / entry_px - 1 - per_side
            cost[i] = per_side
            held[i] = 1
            continue

        if in_pos:
            held[i] = 1
            # Protective exits from the bar after entry, before the close.
            if i > entry_i:
                stops = []
                if exits.stop_loss_pct is not None:
                    stops.append((entry_px * (1 - exits.stop_loss_pct / 100), "stop_loss"))
                if exits.atr_stop_k is not None and not np.isnan(atr_entry):
                    stops.append((entry_px - exits.atr_stop_k * atr_entry, "atr_stop"))
                reason = fill = None
                if stops:
                    level, name = max(stops)  # the tighter stop is the higher level
                    if lows[i] <= level:
                        reason, fill = name, min(opens[i], level)
                if reason is None and exits.trailing_stop_pct is not None:
                    trail = peak * (1 - exits.trailing_stop_pct / 100)
                    if lows[i] <= trail:
                        reason, fill = "trailing_stop", min(opens[i], trail)
                if reason is None and exits.take_profit_pct is not None:
                    target = entry_px * (1 + exits.take_profit_pct / 100)
                    if highs[i] >= target:
                        reason, fill = "take_profit", max(opens[i], target)
                if reason is not None:
                    ret[i] = fill / close[i - 1] - 1 - per_side
                    cost[i] = per_side
                    close_trade(i, fill, reason)
                    in_pos = False
                    held[i] = 0
                    last_exit = i
                    exit_events += 1
                    cooldown_until = i + cooldown
                    continue
            # ordinary carry for a position held since the previous close
            ret[i] = close[i] / close[i - 1] - 1 if i > 0 else 0.0
            peak = max(peak, highs[i])
            held_for = i - entry_i
            if exits.time_stop_sessions is not None and held_for >= exits.time_stop_sessions:
                pending_exit = "time_stop"
            elif exit_sig[i]:
                pending_exit = "indicator_exit"
            continue

        # flat with no pending fill: look for a fresh entry event at this close
        if enter_sig[i] and i + 1 >= cooldown_until and i < n - 1:
            pending_entry = True

    return TickerRun(
        returns=ret,
        costs=cost,
        held=held,
        trades=trades,
        exit_events=exit_events,
        churn_events=churn_events,
    )


def run_strategy(frames: dict[str, pd.DataFrame], entry: list[Rule], exit_: list[Rule],
                 exits: ProtectiveExits, cooldown: int, cost_bps: float, slippage_bps: float,
                 weights: Optional[dict[str, float]] = None):
    """Portfolio daily return and cost, both in percent of equity, plus per-ticker runs.
    Each stock's sleeve is its weight (equal unless `weights` is given, and weights must sum to 1)."""
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    per_side = (cost_bps + slippage_bps) / 10_000
    ret = np.zeros(len(common))
    cost = np.zeros(len(common))
    runs: dict[str, TickerRun] = {}
    for ticker, frame in frames.items():
        w = weights[ticker] if weights else 1.0 / len(frames)
        run = _run_ticker(frame.loc[common], entry, exit_, exits, cooldown, per_side, ticker)
        runs[ticker] = run
        ret += run.returns * w
        cost += run.costs * w
    index = pd.DatetimeIndex(common)
    return pd.Series(ret * 100, index=index), pd.Series(cost * 100, index=index), runs


def basket_returns(frames: dict[str, pd.DataFrame], cost_bps: float, slippage_bps: float,
                   weights: Optional[dict[str, float]] = None) -> pd.Series:
    """Buy-and-hold of the same tickers at their target weights (equal unless given), rebalanced to those
    weights on the first session of each month. Rebalance trades pay the same per-side cost as the strategy."""
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    closes = pd.DataFrame({t: frames[t].loc[common, "close"].astype(float) for t in frames})
    rets = closes.pct_change().fillna(0.0).to_numpy()
    n = closes.shape[1]
    per_side = (cost_bps + slippage_bps) / 10_000
    target_w = np.array([weights[t] for t in closes.columns]) if weights else np.full(n, 1.0 / n)
    dollars = target_w.copy()
    out = []
    prev_month = None
    for k, d in enumerate(closes.index):
        month = (d.year, d.month)
        before = dollars.sum()
        gross = float((dollars * rets[k]).sum() / before)
        dollars = dollars * (1 + rets[k])
        value = dollars.sum()
        cost = 0.0
        if prev_month is not None and month != prev_month:
            target = target_w * value
            one_way = np.abs(dollars - target).sum() / (2 * value)
            cost = 2 * one_way * per_side
            dollars = target
        prev_month = month
        out.append(gross - cost)
    return pd.Series(np.array(out) * 100, index=closes.index)


def strategy_turnover_pct_per_year(runs: dict, weights: dict[str, float], days: int) -> Optional[float]:
    """STB-5: share of the portfolio traded per year, one way. Each entry buys the stock's sleeve and each exit sells it;
    buys and sells are added together and halved, so a stock bought and sold once a year is 100% turnover per year."""
    if days <= 0:
        return None
    traded = 0.0
    for ticker, run in runs.items():
        held = np.asarray(run.held, dtype=float)
        traded += weights[ticker] * float(np.abs(np.diff(held, prepend=0.0)).sum())
    return round(traded / 2 / (days / PERIODS_PER_YEAR) * 100, 1)


def basket_turnover_pct_per_year(frames: dict[str, pd.DataFrame], weights: Optional[dict[str, float]] = None) -> Optional[float]:
    """STB-5: one-way turnover per year of the same basket, rebalanced to its target weights each month."""
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    if not common:
        return None
    closes = pd.DataFrame({t: frames[t].loc[common, "close"].astype(float) for t in frames})
    rets = closes.pct_change().fillna(0.0).to_numpy()
    n = closes.shape[1]
    target_w = np.array([weights[t] for t in closes.columns]) if weights else np.full(n, 1.0 / n)
    dollars = target_w.copy()
    total, prev_month = 0.0, None
    for k, d in enumerate(closes.index):
        month = (d.year, d.month)
        dollars = dollars * (1 + rets[k])
        value = dollars.sum()
        if prev_month is not None and month != prev_month:
            total += np.abs(dollars - target_w * value).sum() / (2 * value)
            dollars = target_w * value
        prev_month = month
    years = len(closes.index) / PERIODS_PER_YEAR
    return round(total / years * 100, 1) if years > 0 else None


def model_portfolio_summary(series: list) -> dict:
    """STB-5: risk and return of the app's own model portfolio, from its growth-of-$10,000 series (see
    build_model_portfolio_series). Each step is one non-overlapping holding period, so the risk figures are annualised
    from the average gap between steps. Withheld until there are 12 months of history, because fewer points would
    give figures that look precise and mean little."""
    if len(series) < 3:
        return {"available": False, "months_of_history": 0.0, "reason": "The model portfolio has no publication history yet."}
    dates = [date.fromisoformat(d) for d, _ in series]
    values = np.array([float(v) for _, v in series])
    span_days = (dates[-1] - dates[0]).days
    months = round(span_days / 30.44, 1)
    if months < 12:
        return {"available": False, "months_of_history": months,
                "reason": f"Only {months} months of model portfolio history so far; at least 12 are needed before its risk figures are shown."}
    rets = np.diff(values) / values[:-1]
    per_year = 365.25 / (span_days / (len(values) - 1))
    years = span_days / 365.25
    vol = float(np.std(rets, ddof=1) * np.sqrt(per_year))
    sharpe = float(np.mean(rets) * per_year / vol) if vol > 0 else None
    peak = np.maximum.accumulate(values)
    return {
        "available": True,
        "months_of_history": months,
        "total_return_pct": _r((values[-1] / values[0] - 1) * 100),
        "cagr_pct": _r(((values[-1] / values[0]) ** (1 / years) - 1) * 100),
        "volatility_pct": _r(vol * 100),
        "max_drawdown_pct": _r(float(((values / peak) - 1).min() * 100)),
        "sharpe": _r(sharpe),
        "worst_period_pct": _r(float(rets.min() * 100)),
        "note": "Figures come from the model portfolio's non-overlapping holding periods, after its assumed trading costs.",
    }


def expected_max_sharpe_by_chance(variants: int, years: float) -> float:
    """Approximate annual Sharpe the best of `variants` no-skill strategies would reach by chance
    (extreme-value approximation). An approximation, labelled as one in the result."""
    if variants <= 1 or years <= 0:
        return 0.0
    return math.sqrt(2 * math.log(variants)) / math.sqrt(years)


def metrics(daily_pct: pd.Series) -> dict:
    series = pd.Series(daily_pct, dtype=float).dropna()
    if series.empty:
        return {"days": 0}
    values = series.tolist()
    years = len(values) / PERIODS_PER_YEAR
    total = cumulative_pct(values)
    cagr = ((1 + total / 100) ** (1 / years) - 1) * 100 if years > 0 and total is not None and total > -100 else None
    vol = float(series.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(series) > 1 else None
    monthly = ((1 + series / 100).groupby(series.index.to_period("M")).prod() - 1) * 100
    return {
        "days": len(values),
        "total_return_pct": _r(total),
        "cagr_pct": _r(cagr),
        "volatility_pct": _r(vol),
        "max_drawdown_pct": _r(max_drawdown_pct(values)),
        "sharpe": _r(sharpe(values, 0.0, PERIODS_PER_YEAR)),
        "worst_month_pct": _r(float(monthly.min())) if len(monthly) else None,
    }


def _r(v: Optional[float], digits: int = 2) -> Optional[float]:
    return None if v is None else round(float(v), digits)


def _check(status: str, label: str, detail: str) -> dict:
    return {"status": status, "label": label, "detail": detail}


def _sensitivity(frames, entry_raw, exit_raw, exits: ProtectiveExits, cooldown, cost_bps, slippage_bps) -> dict:
    """SB-V4: each numeric threshold (rules and percentage stops) moved by the step, one at a time."""
    base: dict[str, float] = {}
    for i, r in enumerate(entry_raw):
        if r.get("field") != REGIME_FIELD:
            base[f"entry[{i}] {r['field']} {r['op']}"] = float(r["value"])
    for i, r in enumerate(exit_raw):
        if r.get("field") != REGIME_FIELD:
            base[f"exit[{i}] {r['field']} {r['op']}"] = float(r["value"])
    for key in ("stop_loss_pct", "trailing_stop_pct", "take_profit_pct"):
        if getattr(exits, key) is not None:
            base[key] = float(getattr(exits, key))

    def run(params: dict) -> Optional[float]:
        e_raw = [dict(r) for r in entry_raw]
        x_raw = [dict(r) for r in exit_raw]
        ex = dict(exits.__dict__)
        for name, value in params.items():
            if name.startswith("entry["):
                i = int(name[len("entry["):name.index("]")])
                e_raw[i]["value"] = value
            elif name.startswith("exit["):
                i = int(name[len("exit["):name.index("]")])
                x_raw[i]["value"] = value
            else:
                ex[name] = value
        e_rules = [Rule.parse(r) for r in e_raw]
        x_rules = [Rule.parse(r) for r in x_raw]
        pe = ProtectiveExits(**{k: v for k, v in ex.items()})
        daily, _, _ = run_strategy(frames, e_rules, x_rules, pe, cooldown, cost_bps, slippage_bps)
        return metrics(daily).get("sharpe")

    return sensitivity(run, base)


def run_backtest(
    frames: dict[str, pd.DataFrame],
    entry_raw: list[dict],
    exit_raw: list[dict],
    benchmark_close: pd.Series,
    variants_tried: int = 1,
    exits_raw: Optional[dict] = None,
    waive_protective_exit: bool = False,
    cooldown: int = DEFAULT_COOLDOWN,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    verdict_benchmark: str = "basket",
    trial_sharpes_daily: Optional[list[float]] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict:
    if not frames:
        raise ValueError("no price history for the chosen tickers")
    if len(frames) > MAX_TICKERS:
        raise ValueError(f"at most {MAX_TICKERS} tickers")
    entry = [Rule.parse(r) for r in entry_raw]
    exit_ = [Rule.parse(r) for r in exit_raw]
    exits = ProtectiveExits.parse(exits_raw)
    if not entry:
        raise ValueError("add at least one entry rule")
    protective = exits.is_protective() or any(r.field for r in exit_)
    if not protective and not waive_protective_exit:
        raise ValueError("add a protective exit (stop, trailing stop, ATR stop or time stop), or waive it")
    if not 0 <= cooldown <= 250:
        raise ValueError("cooldown must be between 0 and 250 sessions")
    if verdict_benchmark not in ("basket", "spy"):
        raise ValueError("verdict benchmark must be 'basket' or 'spy'")

    if weights is not None:
        if set(weights) != set(frames):
            raise ValueError("weights must be given for exactly the tickers tested")
        if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("weights must be zero or more and add up to something above zero")
        total = sum(weights.values())
        weights = {t: w / total for t, w in weights.items()}
    daily, costs_pct, runs = run_strategy(frames, entry, exit_, exits, cooldown, cost_bps, slippage_bps, weights)
    if len(daily) < PERIODS_PER_YEAR // 2:
        raise ValueError("not enough shared history for the chosen tickers")
    gross, _, _ = run_strategy(frames, entry, exit_, exits, cooldown, 0.0, 0.0, weights)

    basket = basket_returns(frames, cost_bps, slippage_bps, weights).reindex(daily.index).dropna()
    spy = (benchmark_close.pct_change().dropna() * 100).reindex(daily.index).dropna()

    split = int(len(daily) * IS_FRACTION)
    is_m, oos_m = metrics(daily.iloc[:split]), metrics(daily.iloc[split:])
    m_strategy, m_gross = metrics(daily), metrics(gross)
    m_basket, m_spy = metrics(basket), metrics(spy)
    # STB-5: turnover. SPY is held, so it never trades; the basket is rebalanced monthly.
    turnover_weights = weights or {t: 1.0 / len(frames) for t in frames}
    m_strategy["turnover_pct_per_year"] = strategy_turnover_pct_per_year(runs, turnover_weights, len(daily))
    m_basket["turnover_pct_per_year"] = basket_turnover_pct_per_year(frames, weights)
    m_spy["turnover_pct_per_year"] = 0.0
    years = len(daily) / PERIODS_PER_YEAR

    all_trades = sorted((t for run in runs.values() for t in run.trades), key=lambda t: t.entry_date)
    round_trips = len(all_trades)
    exit_events = sum(run.exit_events for run in runs.values())
    churn_events = sum(run.churn_events for run in runs.values())
    churn_pct = round(churn_events / exit_events * 100, 1) if exit_events else 0.0
    cost_drag_points = None
    if m_gross.get("cagr_pct") is not None and m_strategy.get("cagr_pct") is not None:
        cost_drag_points = round(m_gross["cagr_pct"] - m_strategy["cagr_pct"], 2)

    s_sh = m_strategy.get("sharpe")
    chance_bar = expected_max_sharpe_by_chance(variants_tried, years)
    walk = walk_forward(daily, basket)
    sens = _sensitivity(frames, entry_raw, exit_raw, exits, cooldown, cost_bps, slippage_bps)
    dsr = deflated_sharpe(daily, list(trial_sharpes_daily or []))
    checks = []
    if is_m.get("sharpe") is not None and oos_m.get("sharpe") is not None:
        if oos_m["sharpe"] < 0 < is_m["sharpe"]:
            checks.append(_check("fail", "Out-of-sample vs in-sample",
                                 "Positive risk-adjusted result in-sample, negative out-of-sample."))
        elif oos_m["sharpe"] - is_m["sharpe"] > OOS_STRONGER_GAP:
            checks.append(_check("caution", "Out-of-sample vs in-sample",
                                 "Out-of-sample is far stronger than in-sample; results likely depend on the recent market, not the rules."))
        else:
            checks.append(_check("pass", "Out-of-sample vs in-sample", "The out-of-sample result is in line with the in-sample one."))
    else:
        checks.append(_check("caution", "Out-of-sample vs in-sample", "Not enough data in one of the periods to compare."))
    if round_trips >= MIN_ROUND_TRIPS:
        checks.append(_check("pass", "Enough trades", f"{round_trips} round trips."))
    else:
        checks.append(_check("caution", "Enough trades",
                             f"{round_trips} round trips; {MIN_ROUND_TRIPS} are needed before the result means much."))
    if s_sh is not None and m_basket.get("sharpe") is not None:
        checks.append(_check("pass" if s_sh > m_basket["sharpe"] else "fail", "Beats same-basket buy-and-hold, risk-adjusted",
                             f"Sharpe {s_sh:.2f} vs {m_basket['sharpe']:.2f}."))
    if s_sh is not None and m_spy.get("sharpe") is not None:
        checks.append(_check("pass" if s_sh > m_spy["sharpe"] else "fail", "Beats SPY, risk-adjusted",
                             f"Sharpe {s_sh:.2f} vs {m_spy['sharpe']:.2f}."))
    if dsr.get("probability") is not None:
        checks.append(_check("pass" if dsr["probability"] >= 0.95 else "caution", "Deflated Sharpe",
                             f"Probability {dsr['probability']:.0%} that the Sharpe is above zero after {dsr['variants']} variant(s) tried."))
    if walk.get("beat_basket_pct") is not None:
        # Fail below 2 in 5 windows, caution below 3 in 5, pass at 3 in 5 or better.
        share = walk["beat_basket_pct"]
        walk_status = "pass" if share >= 60 else "caution" if share >= 40 else "fail"
        checks.append(_check(walk_status, "Walk-forward windows",
                             f"The strategy beat the same stocks in {walk['beat_basket_windows']} of {walk['test_windows']} six-month test windows."))
    if sens.get("widest_swing") is not None:
        checks.append(_check("pass" if sens["widest_swing"] <= 0.5 else "caution", "Stable to small changes",
                             f"Moving each threshold by {sens['step_pct']}% changes the Sharpe by up to {sens['widest_swing']:.2f}."))
    if churn_pct > CHURN_WARN_PCT:
        checks.append(_check("caution", "Churn", f"{churn_pct:.0f}% of exits were followed by a re-entry within {CHURN_WINDOW} sessions."))
    if not protective:
        checks.append(_check("caution", "Protective exit", "No protective exit: losing positions are held until an indicator exit fires."))

    ticker_contrib = {t: round(float(run.returns.sum() * 100 * (weights[t] if weights else 1 / len(runs))), 2) for t, run in runs.items()}
    total_contrib = sum(ticker_contrib.values())
    top = None
    if total_contrib > 0:
        # Share of the total gain. Only meaningful when the total is positive; near-zero totals give silly shares.
        biggest = max(ticker_contrib, key=lambda k: ticker_contrib[k])
        share = ticker_contrib[biggest] / total_contrib
        top = {"ticker": biggest, "share_pct": round(share * 100, 1), "flag": share > 0.5}

    wins = [t.return_pct for t in all_trades if t.return_pct > 0]
    losses = [t.return_pct for t in all_trades if t.return_pct <= 0]
    bench = m_basket if verdict_benchmark == "basket" else m_spy

    def diff(a, b):
        return None if a is None or b is None else round(a - b, 2)

    return {
        "period": {"start": str(daily.index[0].date()), "end": str(daily.index[-1].date()), "sessions": len(daily)},
        "tickers": sorted(frames.keys()),
        "costs": {"cost_bps_per_side": cost_bps, "slippage_bps_per_side": slippage_bps},
        "cooldown_sessions": cooldown,
        "protective_exit": {
            **{k: v for k, v in exits.__dict__.items()},
            "waived": not protective,
        },
        "trades": round_trips,
        "trades_per_year": _r(round_trips / years) if years else None,
        "churn_pct": churn_pct,
        "cost_drag": {"total_costs_pct_of_equity": round(float(costs_pct.sum()), 2), "cagr_points": cost_drag_points},
        "exposure_pct": round(float(np.mean([run.held.mean() for run in runs.values()])) * 100, 1),
        "win_rate_pct": round(len(wins) / round_trips * 100, 1) if round_trips else None,
        "avg_win_pct": round(float(np.mean(wins)), 2) if wins else None,
        "avg_loss_pct": round(float(np.mean(losses)), 2) if losses else None,
        "strategy": m_strategy,
        "basket": m_basket,
        "benchmark_spy": m_spy,
        "in_sample": is_m,
        "out_of_sample": oos_m,
        "per_ticker_contribution": ticker_contrib,
        "top_ticker": top,
        "variants_tried": variants_tried,
        "chance_sharpe_bar": _r(chance_bar),
        "deflated_sharpe": dsr,
        "sharpe_daily": dsr.get("sharpe_daily"),
        "walk_forward": walk,
        "sensitivity": sens,
        "checks": checks,
        "verdict": {
            "benchmark": verdict_benchmark,
            "excess_cagr_pct": diff(m_strategy.get("cagr_pct"), bench.get("cagr_pct")),
            "sharpe_vs_basket": diff(s_sh, m_basket.get("sharpe")),
            "sharpe_vs_spy": diff(s_sh, m_spy.get("sharpe")),
            "beats_benchmark_after_costs": (m_strategy.get("cagr_pct") or 0) > (bench.get("cagr_pct") or 0),
        },
        "trade_log": [
            {
                "ticker": t.ticker,
                "entry_date": str(t.entry_date.date()),
                "entry_price": t.entry_price,
                "exit_date": str(t.exit_date.date()),
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "holding_days": t.holding_days,
                "return_pct": t.return_pct,
            }
            for t in all_trades
        ],
        "equity_curve": {
            "dates": [str(d.date()) for d in daily.index],
            "strategy": np.round(np.cumprod(1 + daily.to_numpy() / 100) * 100, 2).tolist(),
            "basket": np.round(np.cumprod(1 + basket.reindex(daily.index).fillna(0).to_numpy() / 100) * 100, 2).tolist(),
        },
        "state_warnings": state_warnings(entry, exit_),
        "caveats": [
            "Fixed ticker list you chose. Results partly reflect the stocks picked, not only the rules (selection bias).",
            "Scores are not used: the stored score history starts in August 2026.",
            "Model portfolio comparison is not available for the full period: the app's own model portfolio has only a few months of history.",
            "Past results, not a forecast or a recommendation.",
        ],
    }
