"""
Walk-forward validation of the Quant Signal shown on /predict, the Stock
Screener, and the Quant vs Analyst comparison page.

That signal comes from a *recursive* multi-day forecast
(prediction_service.predict_future_prices): a single-day model predicts
tomorrow's return, its own predicted price is appended to the price
history, and the model predicts again from there -- repeated
`horizon_days` times. The single-day model's own backtest (see
model_service.RETURN_SHRINKAGE's comment) only beats a naive "no change"
baseline ~33% of the time even after shrinkage; chaining it recursively
lets any per-step bias (e.g. extrapolating a recent run-up via the
MA5/MA10/Lag1 features) compound across the whole horizon instead of
mean-reverting. This module measures how often the resulting BUY/SELL
call was actually right, instead of assuming it.

Deliberately mirrors prediction_service.predict_backtest_prices' walk-
forward, lookahead-safe pattern (train on `sample.iloc[:-1]`, so the
cutoff row's own Target -- which leaks the very return being predicted --
never reaches the model) and predict_future_prices' recursive-forecast
loop (fixed RSI/MACD/BB_pctB for the horizon, rolling_closes fed forward
from the model's own prior-step output) exactly as production runs it --
otherwise this would validate a different pipeline than the one actually
shown to users.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data_service import get_stock_data
from .feature_service import FEATURE_COLUMNS, build_feature_frame
from .model_service import RETURN_SHRINKAGE, train_model_on_frame

DEFAULT_PERIOD = "3y"
DEFAULT_HORIZON_DAYS = 10
DEFAULT_MIN_TRAIN_ROWS = 252
DEFAULT_STEP_DAYS = 5
DEFAULT_BUY_THRESHOLD = 0.05
DEFAULT_SELL_THRESHOLD = -0.05


def _recursive_forecast(model, sample: pd.DataFrame, horizon_days: int) -> float:
    """Same recursive loop as prediction_service.predict_future_prices,
    seeded from `sample` (real data known as of the cutoff) instead of
    the live full history. Returns only the final day's price -- the
    figure the BUY/HOLD/SELL threshold is actually applied to."""
    rolling_closes = list(sample["Close"].values)
    last_row = sample.iloc[-1]
    fixed_rsi = last_row["RSI"]
    fixed_macd = last_row["MACD"]
    fixed_macd_signal = last_row["MACD_signal"]
    fixed_bb_pctb = last_row["BB_pctB"]

    pred = rolling_closes[-1]
    for _ in range(horizon_days):
        lag1 = rolling_closes[-1]
        ma5 = np.mean(rolling_closes[-5:])
        ma10 = np.mean(rolling_closes[-10:]) if len(rolling_closes) >= 10 else ma5
        return1 = (rolling_closes[-1] - rolling_closes[-2]) / rolling_closes[-2] if len(rolling_closes) >= 2 else 0.0

        X_new = np.array([[lag1, ma5, ma10, return1, fixed_rsi, fixed_macd, fixed_macd_signal, fixed_bb_pctb]])
        pred_return = float(model.predict(X_new)[0]) * RETURN_SHRINKAGE
        pred = rolling_closes[-1] * (1 + pred_return)
        rolling_closes.append(pred)

    return pred


def _signal_for(expected_return: float, buy_threshold: float, sell_threshold: float) -> str:
    if expected_return >= buy_threshold:
        return "BUY"
    if expected_return <= sell_threshold:
        return "SELL"
    return "HOLD"


def backtest_quant_signal(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    min_train_rows: int = DEFAULT_MIN_TRAIN_ROWS,
    step_days: int = DEFAULT_STEP_DAYS,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    tune: bool = False,
) -> Optional[dict]:
    """
    Walk the feature frame forward in `step_days`-sized jumps. At each
    cutoff: train on data strictly before it, run the same recursive
    horizon_days-ahead forecast production uses, derive the BUY/HOLD/SELL
    call, then compare against the *real* close horizon_days later.

    Returns None when there isn't enough history for even one cutoff.
    """
    data = get_stock_data(ticker, period)
    if data.empty:
        return None
    df = build_feature_frame(data)
    if df.empty:
        return None

    n = len(df)
    rows = []
    for i in range(min_train_rows, n - horizon_days + 1, step_days):
        sample = df.iloc[:i]
        train_sample = sample.iloc[:-1]
        if len(train_sample) < min_train_rows - 1:
            continue

        model = train_model_on_frame(train_sample, tune=tune)
        last_close = float(sample["Close"].iloc[-1])
        final_price = _recursive_forecast(model, sample, horizon_days)
        expected_return = (final_price - last_close) / last_close
        signal = _signal_for(expected_return, buy_threshold, sell_threshold)

        actual_price = float(df["Close"].iloc[i - 1 + horizon_days])
        actual_return = (actual_price - last_close) / last_close

        if signal == "BUY":
            correct = actual_return > 0
        elif signal == "SELL":
            correct = actual_return < 0
        else:
            correct = sell_threshold < actual_return < buy_threshold

        rows.append(
            {
                "date": str(df.index[i - 1].date()),
                "signal": signal,
                "expected_return_pct": round(expected_return * 100.0, 2),
                "actual_return_pct": round(actual_return * 100.0, 2),
                "correct": correct,
            }
        )

    if not rows:
        return None

    return {"ticker": ticker, "horizon_days": horizon_days, "rows": rows}


def summarize_quant_signal_backtest(per_ticker_results: list[dict]) -> dict:
    """
    Pools every ticker's rows and reports win rate by signal type -- the
    number that actually answers "does a BUY call here mean anything."
    """
    by_signal: dict[str, list[bool]] = {"BUY": [], "SELL": [], "HOLD": []}
    for result in per_ticker_results:
        if not result:
            continue
        for row in result["rows"]:
            by_signal[row["signal"]].append(row["correct"])

    summary = {}
    for signal, outcomes in by_signal.items():
        summary[signal] = {
            "count": len(outcomes),
            "win_rate_pct": round(100.0 * sum(outcomes) / len(outcomes), 1) if outcomes else None,
        }
    return summary
