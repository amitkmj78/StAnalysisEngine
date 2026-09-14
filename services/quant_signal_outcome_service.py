"""
Pure, zero-I/O evaluation of already-captured Quant Signal calls
(pit_quant_signal) against already-captured real prices (pit_prices) --
the live, out-of-sample counterpart to services/quant_signal_backtest_service's
simulated walk-forward, exactly like signal_publication_service's
evaluate_signal_outcomes_for_date is the live counterpart to
momentum_backtest_service. No lookahead risk: both inputs are historical
snapshots the scheduler already wrote on their own capture day, this just
reads and compares them.

Kept dependency-free (no DB, no network) like prediction_accuracy_service.py
so it stays importable and unit-testable without a live database.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Optional

HOLD_BAND_PCT = 5.0  # matches prediction_service.generate_trading_signal's default thresholds


def compute_quant_signal_outcomes(
    signal_rows: list[dict],
    price_rows: list[dict],
    horizon_days: int,
) -> list[dict]:
    """
    For each captured signal row (ticker, as_of_date, signal,
    expected_return_pct, last_close), finds that ticker's own price
    exactly `horizon_days` *captured* trading days later -- per-ticker,
    not a shared calendar index, since pit_prices' own capture cadence
    already only includes real trading days. A signal whose ticker
    doesn't have that many days of price history yet simply isn't due
    yet and is skipped, not guessed at.

    price_rows: [{"ticker", "price_date", "close"}, ...]
    """
    prices_by_ticker: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for r in price_rows:
        prices_by_ticker[r["ticker"]].append((r["price_date"], r["close"]))
    for series in prices_by_ticker.values():
        series.sort(key=lambda pair: pair[0])

    outcomes = []
    for sig in signal_rows:
        ticker = sig["ticker"]
        as_of_date = sig["as_of_date"]
        entry_price = sig.get("last_close")
        if not entry_price:
            continue

        series = prices_by_ticker.get(ticker)
        if not series:
            continue

        entry_idx = next((i for i, (d, _) in enumerate(series) if d == as_of_date), None)
        if entry_idx is None:
            continue

        exit_idx = entry_idx + horizon_days
        if exit_idx >= len(series):
            continue  # not due yet

        exit_date, exit_price = series[exit_idx]
        realized_return_pct = (exit_price - entry_price) / entry_price * 100.0
        signal = sig["signal"]

        if signal == "BUY":
            correct = realized_return_pct > 0
        elif signal == "SELL":
            correct = realized_return_pct < 0
        else:
            correct = -HOLD_BAND_PCT < realized_return_pct < HOLD_BAND_PCT

        outcomes.append(
            {
                "ticker": ticker,
                "as_of_date": as_of_date,
                "signal": signal,
                "expected_return_pct": sig["expected_return_pct"],
                "entry_price": entry_price,
                "horizon_days": horizon_days,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "realized_return_pct": round(realized_return_pct, 3),
                "correct": correct,
            }
        )

    return outcomes


def summarize_quant_signal_outcomes(outcomes: list[dict]) -> dict:
    """Win rate by signal type, pooled across every ticker/date -- the
    number that answers "does a BUY call here mean anything," not a
    per-ticker breakdown (per-ticker samples are usually too small,
    exactly what this project's own MIN_VERIFIED_FOR_RECOMMENDATION
    convention in prediction_accuracy_service.py already accounts for)."""
    by_signal: dict[str, list[bool]] = {"BUY": [], "SELL": [], "HOLD": []}
    for o in outcomes:
        by_signal.setdefault(o["signal"], []).append(o["correct"])

    summary = {}
    for signal, outcomes_for_signal in by_signal.items():
        summary[signal] = {
            "count": len(outcomes_for_signal),
            "win_rate_pct": (
                round(100.0 * sum(outcomes_for_signal) / len(outcomes_for_signal), 1)
                if outcomes_for_signal
                else None
            ),
        }
    return summary
