"""Scan the starting templates on a seeded random sample of S&P 500 stocks.

Output is two short lists of CANDIDATES to study, not recommendations:
  - short-term: strategies whose measured average hold is about a month or less (or that trade often)
  - long-term: strategies whose measured average hold is over about six months, ranked against the S&P 500

Every template is measured on the later 30% of dates (the out-of-sample period), which the rules were not
picked on. Holding time comes from the trades that were actually made in that period, so the same template
can fall into a different group under different settings.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from services.strategy_engine import (
    IS_FRACTION,
    PERIODS_PER_YEAR,
    ProtectiveExits,
    Rule,
    basket_returns,
    metrics,
    run_strategy,
)
from services.strategy_robustness import deflated_sharpe

SCAN_SIZE = 20
SCAN_COST_BPS = 10.0
SCAN_SLIPPAGE_BPS = 5.0
SCAN_COOLDOWN = 5

SHORT_MAX_HOLD_DAYS = 21            # about one month of trading days
SHORT_MIN_TRADES_PER_YEAR = 12
SHORT_TRADE_COUNT_MAX_HOLD_DAYS = 42   # trade count only counts as short-term when holds average under about two months
LONG_MIN_HOLD_DAYS = 126            # about six months of trading days
MIN_TRADES_FOR_SHORT = 30
TOP_PER_GROUP = 5
DOWNTURN_DRAWDOWN_PCT = -15.0       # the S&P 500 must fall at least this far inside the window
ROUND_TRIP_COST_PCT = 2 * (SCAN_COST_BPS + SCAN_SLIPPAGE_BPS) / 100   # 0.30% per round trip


@dataclass(frozen=True)
class Template:
    key: str
    name: str
    entry: tuple
    exit: tuple
    exits: dict


TEMPLATES: tuple[Template, ...] = (
    Template("trend", "Trend follow (200-day)",
             ({"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0},),
             ({"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": 0},),
             {"trailing_stop_pct": 15.0}),
    Template("pullback", "Pullback in uptrend",
             ({"field": "rsi_14", "op": "crosses_above", "value": 30}, {"field": "close_vs_sma_200_pct", "op": ">", "value": 0}),
             ({"field": "rsi_14", "op": ">", "value": 70},),
             {"trailing_stop_pct": 10.0}),
    Template("breakout", "52-week high breakout",
             ({"field": "dist_52w_high_pct", "op": "crosses_above", "value": -2},),
             ({"field": "dist_52w_high_pct", "op": "<", "value": -10},),
             {"stop_loss_pct": 8.0}),
    Template("cross", "20/50-day cross",
             ({"field": "sma_20_vs_50_pct", "op": "crosses_above", "value": 0},),
             ({"field": "sma_20_vs_50_pct", "op": "crosses_below", "value": 0},),
             {"trailing_stop_pct": 12.0}),
    Template("lowvol", "Calm uptrend (low volatility)",
             ({"field": "atr_14_pct", "op": "<", "value": 2}, {"field": "close_vs_sma_50_pct", "op": "crosses_above", "value": 0}),
             ({"field": "close_vs_sma_50_pct", "op": "crosses_below", "value": 0},),
             {"stop_loss_pct": 6.0}),
    Template("volume", "Volume-confirmed trend",
             ({"field": "volume_vs_20d_pct", "op": "crosses_above", "value": 50}, {"field": "close_vs_sma_50_pct", "op": ">", "value": 0}),
             (),
             {"stop_loss_pct": 7.0, "time_stop_sessions": 20}),
    Template("regime", "Risk-On trend",
             ({"field": "regime", "op": "is", "value": "Risk-On"}, {"field": "close_vs_sma_50_pct", "op": "crosses_above", "value": 0}),
             ({"field": "regime", "op": "is", "value": "Risk-Off"},),
             {"stop_loss_pct": 10.0}),
    Template("rsi", "RSI momentum",
             ({"field": "rsi_14", "op": "crosses_above", "value": 50},),
             ({"field": "rsi_14", "op": ">", "value": 60},),
             {"trailing_stop_pct": 10.0}),
    # Long-horizon templates: wide stops and slow exits, so the measured hold is months, not weeks.
    Template("long_trend", "Long trend hold (200-day, wide stop)",
             ({"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0},),
             ({"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": 0},),
             {"trailing_stop_pct": 30.0}),
    Template("long_strength", "Long strength hold (near 52-week high)",
             ({"field": "dist_52w_high_pct", "op": "crosses_above", "value": -5},),
             ({"field": "dist_52w_high_pct", "op": "<", "value": -25},),
             {"trailing_stop_pct": 30.0}),
    Template("long_slow", "Long trend, slow exit (200-day, no stop)",
             ({"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0},),
             ({"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": -5},),
             {"stop_loss_pct": 40.0}),
)


def pick_sample(universe: list[str], seed: int, size: int = SCAN_SIZE) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(universe), min(size, len(universe))))


def _total_pct(series_pct: pd.Series) -> float:
    return float(((1 + series_pct / 100).prod() - 1) * 100) if len(series_pct) else 0.0


def _max_dd(daily_pct: pd.Series) -> Optional[float]:
    if daily_pct.empty:
        return None
    equity = (1 + daily_pct / 100).cumprod()
    return float(((equity / equity.cummax() - 1) * 100).min())


def _holding_class(avg_hold_days: Optional[float], trades_per_year: float) -> str:
    if avg_hold_days is None:
        return "none"
    # A high trade count only makes a strategy short-term when its holds are short too. A strategy that holds
    # for months but re-enters often is not short-term.
    if avg_hold_days < SHORT_MAX_HOLD_DAYS or (trades_per_year > SHORT_MIN_TRADES_PER_YEAR and avg_hold_days < SHORT_TRADE_COUNT_MAX_HOLD_DAYS):
        return "short"
    if avg_hold_days <= LONG_MIN_HOLD_DAYS:
        return "swing"
    return "long"


def _warnings(is_excess: float, oos_excess: float) -> list[str]:
    out = []
    if oos_excess <= 0:
        out.append("Did not beat holding the same stocks on the later dates.")
    if is_excess > 0 and oos_excess < 0:
        out.append("Looked good on earlier dates and lost on later ones.")
    if oos_excess > 0 and is_excess < 0:
        out.append("Lagged on earlier dates but did better later; the recent market may explain it.")
    return out


def _profile(r: dict) -> str:
    if r["holding_class"] == "short":
        return (f"Short-term · average hold {r['avg_hold_days']:.0f} trading days · {r['trades_oos']} trades out-of-sample · "
                f"{r['oos_return_after_costs_pct']:+.1f}% after costs vs {r['oos_return_vs_basket_pct']:+.1f}% for holding the same stocks")
    if r["holding_class"] == "long":
        dd = "smaller" if (r["full_max_drawdown_pct"] or 0) >= (r["spy_full_max_drawdown_pct"] or 0) else "larger"
        months = (r["avg_hold_days"] or 0) / 21
        return (f"Long-term · holds ~{months:.0f} months · {r['full_cagr_vs_spy_pct']:+.1f}%/yr vs the S&P 500 over five years · {dd} drawdowns")
    if r["holding_class"] == "swing":
        return f"Swing · average hold {r['avg_hold_days']:.0f} trading days · {r['trades_oos']} trades out-of-sample"
    return "No trades out-of-sample"


def scan(
    frames: dict[str, pd.DataFrame],
    bench_close: pd.Series,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Runs every template on the same stocks and dates, measures each on the later 30% of dates, and sorts
    the templates into short-term and long-term lists."""
    total = len(TEMPLATES)
    runs_by_key = {}
    for i, t in enumerate(TEMPLATES, 1):
        entry = [Rule.parse(dict(r)) for r in t.entry]
        exit_rules = [Rule.parse(dict(r)) for r in t.exit]
        exits = ProtectiveExits(**t.exits)
        daily, _, runs = run_strategy(frames, entry, exit_rules, exits, SCAN_COOLDOWN, SCAN_COST_BPS, SCAN_SLIPPAGE_BPS)
        runs_by_key[t.key] = (t, daily, runs)
        if progress:
            progress(i, total)

    dates = next(iter(runs_by_key.values()))[1].index
    basket_all = basket_returns(frames, SCAN_COST_BPS, SCAN_SLIPPAGE_BPS).reindex(dates).dropna()
    spy_all = (bench_close.pct_change().dropna() * 100).reindex(basket_all.index).dropna()
    dates = spy_all.index
    split = int(len(dates) * IS_FRACTION)
    is_dates, oos_dates = dates[:split], dates[split:]
    oos_years = len(oos_dates) / PERIODS_PER_YEAR
    spy_oos_m = metrics(spy_all.reindex(oos_dates).dropna())
    spy_full_m = metrics(spy_all)
    downturn_included = (_max_dd(spy_all) or 0) <= DOWNTURN_DRAWDOWN_PCT

    trial_sharpes = []
    for _, daily, _ in runs_by_key.values():
        d = daily.reindex(dates).dropna() / 100
        if len(d) > 1 and d.std(ddof=1) > 0:
            trial_sharpes.append(float(d.mean() / d.std(ddof=1)))

    rows = []
    for key, (t, daily, runs) in runs_by_key.items():
        daily = daily.reindex(dates).dropna()
        basket = basket_all.reindex(daily.index).dropna()
        strat_oos = daily.reindex(oos_dates).dropna()
        basket_oos = basket.reindex(oos_dates).dropna()
        oos_trades = [tr for run in runs.values() for tr in run.trades if pd.Timestamp(tr.exit_date) >= oos_dates[0]]
        n_trades = len(oos_trades)
        avg_hold = float(np.mean([tr.holding_days for tr in oos_trades])) if oos_trades else None
        trades_per_year = n_trades / oos_years if oos_years else 0.0
        net = [tr.return_pct - ROUND_TRIP_COST_PCT for tr in oos_trades]
        oos_m = metrics(strat_oos)
        full_m = metrics(daily)
        full_cagr = full_m.get("cagr_pct")
        spy_full_cagr = spy_full_m.get("cagr_pct")
        is_excess = _total_pct(daily.reindex(is_dates).dropna()) - _total_pct(basket.reindex(is_dates).dropna())
        oos_excess = _total_pct(strat_oos) - _total_pct(basket_oos)
        dsr = deflated_sharpe(daily, trial_sharpes)
        oos_cagr = oos_m.get("cagr_pct")
        spy_cagr = spy_oos_m.get("cagr_pct")
        rows.append({
            "key": key,
            "name": t.name,
            "holding_class": _holding_class(avg_hold, trades_per_year),
            "avg_hold_days": round(avg_hold, 1) if avg_hold is not None else None,
            "trades_oos": n_trades,
            "trades_per_year": round(trades_per_year, 1),
            "win_rate_pct": round(sum(1 for x in net if x > 0) / n_trades * 100, 1) if n_trades else None,
            "avg_trade_net_pct": round(float(np.mean(net)), 2) if net else None,
            "oos_return_after_costs_pct": round(_total_pct(strat_oos), 2),
            "oos_return_vs_basket_pct": round(oos_excess, 2),
            "oos_cagr_pct": oos_cagr,
            "oos_cagr_vs_spy_pct": round(oos_cagr - spy_cagr, 2) if oos_cagr is not None and spy_cagr is not None else None,
            "oos_max_drawdown_pct": oos_m.get("max_drawdown_pct"),
            "spy_oos_max_drawdown_pct": spy_oos_m.get("max_drawdown_pct"),
            "full_cagr_pct": full_cagr,
            "full_cagr_vs_spy_pct": round(full_cagr - spy_full_cagr, 2) if full_cagr is not None and spy_full_cagr is not None else None,
            "full_max_drawdown_pct": full_m.get("max_drawdown_pct"),
            "spy_full_max_drawdown_pct": spy_full_m.get("max_drawdown_pct"),
            "deflated_probability": dsr.get("probability"),
            "warnings": _warnings(is_excess, oos_excess),
        })

    for r in rows:
        r["profile"] = _profile(r)

    short = sorted(
        (r for r in rows if r["holding_class"] == "short" and r["trades_oos"] >= MIN_TRADES_FOR_SHORT),
        key=lambda r: r["oos_return_after_costs_pct"], reverse=True,
    )
    long_pool = sorted(
        (r for r in rows
         if r["holding_class"] == "long"
         and r["full_cagr_vs_spy_pct"] is not None and r["full_cagr_vs_spy_pct"] >= 0
         and (r["full_max_drawdown_pct"] or 0) >= (r["spy_full_max_drawdown_pct"] or 0)),
        key=lambda r: r["full_cagr_vs_spy_pct"], reverse=True,
    )
    if not downturn_included:
        long_message = "The test window has no major downturn in the S&P 500, so long-term results are not ranked."
        long_candidates = []
    else:
        long_candidates = long_pool[:TOP_PER_GROUP]
        long_message = None if long_candidates else "No strong long-term candidates in this sample."

    rows.sort(key=lambda r: r["oos_return_after_costs_pct"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return {
        "sample_size": len(frames),
        "tickers": sorted(frames),
        "variants_tried": len(TEMPLATES),
        "period": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "out_of_sample": {"start": str(oos_dates[0].date()), "years": round(oos_years, 2)},
        "benchmark": {"name": "S&P 500 (SPY)", "oos_cagr_pct": spy_oos_m.get("cagr_pct"),
                      "oos_max_drawdown_pct": spy_oos_m.get("max_drawdown_pct")},
        "downturn_in_test_window": downturn_included,
        "groups": {
            "short_term": {
                "ranked_by": "Out-of-sample return after costs",
                "minimum": f"At least {MIN_TRADES_FOR_SHORT} trades out-of-sample",
                "candidates": short[:TOP_PER_GROUP],
                "message": None if short else "No strong short-term candidates in this sample.",
            },
            "long_term": {
                "ranked_by": "Five-year CAGR vs the S&P 500",
                "minimum": "Matches or beats the S&P 500 over the full five years with a drawdown no deeper, in a window with a major downturn",
                "candidates": long_candidates,
                "message": long_message,
            },
        },
        "candidates": rows,
        "note": (
            "Candidates to study, not recommendations. Measured on the last 30% of dates, which the rules were not "
            "picked on. Short-term results include an estimate of trading costs and slippage. Frequent trading is often "
            "taxed as short-term gains; this is not tax advice. The chance figure accounts for all templates tried in this scan."
        ),
    }
