"""Scan the starting templates on a seeded random sample of S&P 500 stocks.

Output is a ranked list of CANDIDATES to study, not recommendations. Ranking uses the out-of-sample
period (the last 30% of dates), because the in-sample period is where a rule is most likely to look
good by chance. Each candidate carries its deflated Sharpe, computed over all templates in the scan,
so the count of variants tried is part of the result.
"""

import random
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from services.strategy_engine import (
    IS_FRACTION,
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
)


def pick_sample(universe: list[str], seed: int, size: int = SCAN_SIZE) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(universe), min(size, len(universe))))


def _total_pct(series_pct: pd.Series) -> float:
    return float(((1 + series_pct / 100).prod() - 1) * 100) if len(series_pct) else 0.0


def scan(
    frames: dict[str, pd.DataFrame],
    bench_close: pd.Series,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Runs every template on the same stocks and dates. Returns candidates ranked by out-of-sample excess return."""
    rows = []
    daily_by_key = {}
    total = len(TEMPLATES)
    basket_all = basket_returns(frames, SCAN_COST_BPS, SCAN_SLIPPAGE_BPS)
    for i, t in enumerate(TEMPLATES, 1):
        entry = [Rule.parse(dict(r)) for r in t.entry]
        exit_rules = [Rule.parse(dict(r)) for r in t.exit]
        exits = ProtectiveExits(**t.exits)
        daily, _, _ = run_strategy(frames, entry, exit_rules, exits, SCAN_COOLDOWN, SCAN_COST_BPS, SCAN_SLIPPAGE_BPS)
        basket = basket_all.reindex(daily.index).dropna()
        daily = daily.reindex(basket.index).dropna()
        daily_by_key[t.key] = (t, daily, basket)
        if progress:
            progress(i, total)

    trial_sharpes = []
    for t, daily, _ in daily_by_key.values():
        values = daily.to_numpy() / 100
        if len(values) > 1 and values.std(ddof=1) > 0:
            trial_sharpes.append(float(values.mean() / values.std(ddof=1)))

    for key, (t, daily, basket) in daily_by_key.items():
        split = int(len(daily) * IS_FRACTION)
        is_strat, oos_strat = daily.iloc[:split], daily.iloc[split:]
        is_basket, oos_basket = basket.iloc[:split], basket.iloc[split:]
        oos_excess = _total_pct(oos_strat) - _total_pct(oos_basket)
        is_excess = _total_pct(is_strat) - _total_pct(is_basket)
        dsr = deflated_sharpe(daily, trial_sharpes)
        oos_m = metrics(oos_strat)
        rows.append({
            "key": key,
            "name": t.name,
            "oos_excess_return_pct": round(oos_excess, 2),
            "in_sample_excess_return_pct": round(is_excess, 2),
            "oos_sharpe": oos_m.get("sharpe"),
            "deflated_probability": dsr.get("probability"),
            "warnings": _warnings(is_excess, oos_excess),
        })

    rows.sort(key=lambda r: r["oos_excess_return_pct"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return {
        "sample_size": len(frames),
        "tickers": sorted(frames),
        "variants_tried": len(TEMPLATES),
        "period": {"start": str(next(iter(daily_by_key.values()))[1].index[0].date()),
                   "end": str(next(iter(daily_by_key.values()))[1].index[-1].date())},
        "candidates": rows,
        "note": (
            "Candidates to study, not recommendations. Ranked on the last 30% of dates, which the rules were not "
            "picked on. The chance-of-luck figure accounts for all templates tried in this scan."
        ),
    }


def _warnings(is_excess: float, oos_excess: float) -> list[str]:
    out = []
    if oos_excess <= 0:
        out.append("Did not beat holding the same stocks on the later dates.")
    if is_excess > 0 and oos_excess < 0:
        out.append("Looked good on earlier dates and lost on later ones.")
    if oos_excess > 0 and is_excess < 0:
        out.append("Lagged on earlier dates but did better later; the recent market may explain it.")
    return out

