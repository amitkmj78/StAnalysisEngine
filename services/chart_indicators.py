"""Chart indicators for the stock page (CHT-3). Pure functions over pandas
Series, so each one can be checked against a hand-worked value.

Conventions, stated because they change the numbers:
- EMA uses span-based smoothing (alpha = 2 / (n + 1)), the common charting default.
- RSI and ATR use Wilder smoothing (alpha = 1/n), the standard definition and the one TradingView uses.
- Bollinger bands use the population standard deviation (ddof = 0).
- VWAP is anchored to the first bar of the series; the daily chart has no session reset.
- OBV starts at zero on the first bar.
"""

import numpy as np
import pandas as pd


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI. Rising-only history gives 100, falling-only gives 0.
    The first bar has no prior close, so its change counts as zero (the
    convention used by the standard reference implementation)."""
    change = close.diff().fillna(0.0)
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0).where(~((avg_gain == 0) & (avg_loss == 0)))
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3
    cum_pv = (typical * volume).cumsum()
    cum_v = volume.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).rename("vwap")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ATR: seeded with the simple mean of the first n true ranges, then
    atr_i = (atr_{i-1} * (n - 1) + TR_i) / n. This is the chart's definition; the
    trading agent's services/agent/indicators.atr uses a simple mean and is a
    separate function, left unchanged."""
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    values = true_range.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) >= n:
        out[n - 1] = values[:n].mean()
        for i in range(n, len(values)):
            out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return pd.Series(out, index=close.index, name="atr")


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")
