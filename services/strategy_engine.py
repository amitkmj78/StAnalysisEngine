"""STB-1/2/3/4/5: no-code strategy engine for backtests.

A strategy is a list of entry rules (all must hold) and a list of exit rules (any one closes
the position). Each rule compares one allowed field to a value, chosen from dropdowns in the UI.

Execution, stated so the numbers can be checked:
  - Rules are read at each day's close. Orders fill at the NEXT day's open (no same-bar fills).
  - Each trade pays `cost_bps` per side plus `slippage_bps` per side.
  - Long only. The portfolio gives each ticker an equal 1/N sleeve; idle sleeves earn nothing.
  - Daily returns, 252 sessions a year, risk-free rate zero.

Out-of-sample (STB-3): the first `IS_FRACTION` of dates are "in-sample" and the rest are
"out-of-sample". The rules are not refitted between the two, so the split only shows whether
the result holds on dates after the ones it was looked at. Refitting on the first part and
testing on the second is a follow-up.

Overfitting (STB-4): the caller passes how many rule variants this user has tried. The result
flags a likely-chance result when many variants were tried, or when the out-of-sample result
is much weaker than the in-sample one.

Data limits (STB-2): the price history is whatever the caller supplies (yfinance, several years).
The app's stored scores begin in August 2026, so score fields are not offered here. Membership
is a user-chosen list, not the historical index, so survivorship bias is not removed.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from services.backtest_engine import cumulative_pct, max_drawdown_pct, sharpe
from services.chart_indicators import rsi, sma

PERIODS_PER_YEAR = 252
IS_FRACTION = 0.7
DEFAULT_COST_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
MAX_TICKERS = 20
MANY_VARIANTS = 10

NUMERIC_FIELDS = {
    "rsi_14": "RSI (14)",
    "close_vs_sma_50_pct": "Price vs 50-day average (%)",
    "close_vs_sma_200_pct": "Price vs 200-day average (%)",
    "sma_20_vs_50_pct": "20-day vs 50-day average (%)",
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
        field, op, value = raw.get("field"), raw.get("op"), raw.get("value")
        if field == REGIME_FIELD:
            if op not in CATEGORY_OPS or value not in REGIME_LABELS:
                raise ValueError("regime rules compare to one of the regime labels with 'is' or 'is_not'")
        elif field in NUMERIC_FIELDS:
            if op not in NUMERIC_OPS:
                raise ValueError(f"{field} supports >, <, >=, <=, crosses_above, crosses_below")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{field} needs a number")
        else:
            raise ValueError(f"unknown field: {field}")
        return Rule(field, op, value)


def feature_frame(prices: pd.DataFrame, regime_by_date: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """prices: columns Open, Close (DatetimeIndex). Returns the fields rules can use, per date."""
    close = prices["Close"].astype(float)
    opens = prices["Open"].astype(float)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    out = pd.DataFrame(index=prices.index)
    out["open"] = opens
    out["close"] = close
    out["rsi_14"] = rsi(close, 14)
    out["close_vs_sma_50_pct"] = (close / sma50 - 1) * 100
    out["close_vs_sma_200_pct"] = (close / sma200 - 1) * 100
    out["sma_20_vs_50_pct"] = (sma(close, 20) / sma50 - 1) * 100
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


def _all(frame: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for r in rules:
        mask &= rule_mask(frame, r).fillna(False).astype(bool)
    return mask


def _any(frame: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for r in rules:
        mask |= rule_mask(frame, r).fillna(False).astype(bool)
    return mask


def ticker_position(frame: pd.DataFrame, entry: list[Rule], exit_: list[Rule]) -> pd.DataFrame:
    """Position state per day. A signal at day t's close changes the position at day t+1's open.
    Exits win on a day when both fire. Returns columns: held (0/1 during the day), trade (+1 buy,
    -1 sell, 0 none), which fills at that day's open."""
    enter = _all(frame, entry) if entry else pd.Series(False, index=frame.index)
    leave = _any(frame, exit_) if exit_ else pd.Series(False, index=frame.index)
    held = np.zeros(len(frame), dtype=int)
    trade = np.zeros(len(frame), dtype=int)
    state = 0
    for i in range(len(frame)):
        if i > 0:
            if state == 1 and leave.iloc[i - 1]:
                state = 0
                trade[i] = -1
            elif state == 0 and enter.iloc[i - 1] and not leave.iloc[i - 1]:
                state = 1
                trade[i] = 1
        held[i] = state
    return pd.DataFrame({"held": held, "trade": trade}, index=frame.index)


def simulate(
    frames: dict[str, pd.DataFrame],
    entry: list[Rule],
    exit_: list[Rule],
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.Series:
    """Equal-weight portfolio daily return in percent, on the dates all tickers share."""
    if not frames:
        return pd.Series(dtype=float)
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    n = len(frames)
    per_side = (cost_bps + slippage_bps) / 10_000
    total = pd.Series(0.0, index=common)
    for frame in frames.values():
        f = frame.loc[common]
        pos = ticker_position(f, entry, exit_)
        held = pos["held"].to_numpy()
        trade = pos["trade"].to_numpy()
        close = f["close"].to_numpy(dtype=float)
        opens = f["open"].to_numpy(dtype=float)
        ret = np.zeros(len(common))
        for i in range(1, len(common)):
            if held[i] and held[i - 1]:
                ret[i] = close[i] / close[i - 1] - 1
            elif held[i] and not held[i - 1]:
                ret[i] = close[i] / opens[i] - 1
            elif not held[i] and held[i - 1]:
                ret[i] = opens[i] / close[i - 1] - 1
            if trade[i] != 0:
                ret[i] -= per_side
        total += pd.Series(ret, index=common) / n
    return total * 100


def trade_count(frames: dict[str, pd.DataFrame], entry: list[Rule], exit_: list[Rule]) -> int:
    count = 0
    for frame in frames.values():
        count += int((ticker_position(frame, entry, exit_)["trade"] != 0).sum())
    return count


def metrics(daily_pct: pd.Series) -> dict:
    """daily_pct: daily returns in percent, indexed by date. Worst month compounds the days in each month."""
    series = pd.Series(daily_pct, dtype=float).dropna()
    if series.empty:
        return {"days": 0}
    values = series.tolist()
    years = len(values) / PERIODS_PER_YEAR
    total = cumulative_pct(values)
    cagr = ((1 + total / 100) ** (1 / years) - 1) * 100 if years > 0 and total is not None and total > -100 else None
    vol = float(series.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(series) > 1 else None
    dd = max_drawdown_pct(values)
    sh = sharpe(values, 0.0, PERIODS_PER_YEAR)
    monthly = ((1 + series / 100).groupby(series.index.to_period("M")).prod() - 1) * 100
    return {
        "days": len(values),
        "total_return_pct": _r(total),
        "cagr_pct": _r(cagr),
        "volatility_pct": _r(vol),
        "max_drawdown_pct": _r(dd),
        "sharpe": _r(sh),
        "worst_month_pct": _r(float(monthly.min())) if len(monthly) else None,
    }


def _r(v: Optional[float], digits: int = 2) -> Optional[float]:
    return None if v is None else round(float(v), digits)


def run_backtest(
    frames: dict[str, pd.DataFrame],
    entry_raw: list[dict],
    exit_raw: list[dict],
    benchmark_close: pd.Series,
    variants_tried: int = 1,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    if not frames:
        raise ValueError("no price history for the chosen tickers")
    if len(frames) > MAX_TICKERS:
        raise ValueError(f"at most {MAX_TICKERS} tickers")
    entry = [Rule.parse(r) for r in entry_raw]
    exit_ = [Rule.parse(r) for r in exit_raw]
    if not entry:
        raise ValueError("add at least one entry rule")
    if not exit_:
        raise ValueError("add at least one exit rule")

    daily = simulate(frames, entry, exit_, cost_bps, slippage_bps)
    if len(daily) < PERIODS_PER_YEAR // 2:
        raise ValueError("not enough shared history for the chosen tickers")
    split = int(len(daily) * IS_FRACTION)
    is_part, oos_part = daily.iloc[:split], daily.iloc[split:]

    bench_returns = benchmark_close.pct_change().dropna() * 100
    bench_on_dates = bench_returns.reindex(daily.index).dropna()

    is_m, oos_m = metrics(is_part), metrics(oos_part)
    warnings: list[str] = []
    if variants_tried >= MANY_VARIANTS:
        warnings.append(
            f"You have tried {variants_tried} rule variants. With that many tries, a good-looking result can appear by chance."
        )
    if is_m.get("sharpe") is not None and oos_m.get("sharpe") is not None and is_m["sharpe"] > 0 and oos_m["sharpe"] < is_m["sharpe"] / 2:
        warnings.append("The out-of-sample Sharpe ratio is less than half the in-sample one. The result may not hold on later dates.")
    if is_m.get("total_return_pct") is not None and oos_m.get("total_return_pct") is not None and oos_m["total_return_pct"] < 0 < is_m["total_return_pct"]:
        warnings.append("The strategy made money in-sample but lost money out-of-sample.")

    return {
        "period": {"start": str(daily.index[0]), "end": str(daily.index[-1]), "sessions": len(daily)},
        "tickers": sorted(frames.keys()),
        "costs": {"cost_bps_per_side": cost_bps, "slippage_bps_per_side": slippage_bps},
        "trades": trade_count(frames, entry, exit_),
        "trades_per_year": _r(trade_count(frames, entry, exit_) / (len(daily) / PERIODS_PER_YEAR)),
        "strategy": metrics(daily),
        "in_sample": is_m,
        "out_of_sample": oos_m,
        "benchmark_spy": metrics(bench_on_dates),
        "variants_tried": variants_tried,
        "warnings": warnings,
        "caveats": [
            "Fixed ticker list you chose. It is not the historical index, so stocks that left the index are missing (survivorship bias).",
            "Scores are not used: the stored score history starts in August 2026.",
            "Model portfolio comparison is not available for the full period: the app's own model portfolio has only a few months of history.",
            "Past results, not a forecast or a recommendation.",
        ],
    }
