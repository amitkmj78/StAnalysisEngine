"""Pure indicator helpers for the trading agent (no I/O).

Inputs are pandas Series/DataFrames already fetched by the caller, so
these run identically in tests and in the scheduled job.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility_pct(closes: pd.Series, window: int) -> Optional[float]:
    """Std of daily % returns over the last `window` returns, annualized."""
    if closes is None or len(closes) < window + 1:
        return None
    returns = closes.pct_change().dropna().tail(window)
    if len(returns) < 2:
        return None
    vol = float(np.std(returns.values, ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
    return round(vol, 4) if math.isfinite(vol) else None


def average_dollar_volume(closes: pd.Series, volumes: pd.Series, window: int) -> Optional[float]:
    """Mean of close * volume over the last `window` sessions."""
    if closes is None or volumes is None or len(closes) < window or len(volumes) < window:
        return None
    dollar = (closes * volumes).tail(window)
    if dollar.isna().any():
        return None
    return float(dollar.mean())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Optional[float]:
    """Simple mean of true range over `window` sessions (same convention as entry_strategy_service)."""
    if min(len(high), len(low), len(close)) < window + 1:
        return None
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    value = true_range.tail(window).mean()
    return float(value) if math.isfinite(value) else None


def sma(closes: pd.Series, window: int) -> Optional[float]:
    if closes is None or len(closes) < window:
        return None
    value = float(closes.tail(window).mean())
    return value if math.isfinite(value) else None
