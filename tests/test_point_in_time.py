"""NFR-1: point-in-time. A value on date d may use prices up to and including d, never later ones.

Each check computes a feature on the full history and on a history cut off earlier. Every value
on the shared dates must match. If a feature peeks at future prices, the values will differ and the
test fails.
"""

import numpy as np
import pandas as pd
import pytest

from services.chart_indicators import atr, bollinger, ema, macd, obv, rsi, sma, vwap
from services.strategy_engine import feature_frame


def _history(n=500, seed=11):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    idx = pd.bdate_range("2019-01-01", periods=n)
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame(
        {"Open": close * (1 + rng.normal(0, 0.002, n)), "High": high, "Low": low,
         "Close": close, "Volume": rng.integers(1_000, 10_000, n).astype(float)},
        index=idx,
    )


CUT = 380  # the shortened history ends here; every check compares dates up to this point


def _assert_same_on_shared_dates(full: pd.Series, cut: pd.Series):
    shared = cut.index
    a = full.loc[shared].to_numpy(dtype=float)
    b = cut.to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-12), "a value uses data from after its own date"


def test_strategy_features_do_not_use_future_prices():
    hist = _history()
    full = feature_frame(hist)
    cut = feature_frame(hist.iloc[:CUT])
    for column in ("rsi_14", "close_vs_sma_50_pct", "close_vs_sma_200_pct", "sma_20_vs_50_pct",
                   "dist_52w_high_pct", "volume_vs_20d_pct", "atr_14", "atr_14_pct"):
        _assert_same_on_shared_dates(full[column], cut[column])


def test_chart_indicators_do_not_use_future_prices():
    hist = _history()
    close, high, low, vol = hist["Close"], hist["High"], hist["Low"], hist["Volume"]
    cut = slice(0, CUT)
    checks = {
        "sma": sma(close, 20),
        "ema": ema(close, 20),
        "rsi": rsi(close, 14),
        "atr": atr(high, low, close, 14),
        "obv": obv(close, vol),
        "vwap": vwap(high, low, close, vol),
    }
    for name, full_series in checks.items():
        short = {
            "sma": lambda: sma(close.iloc[cut], 20),
            "ema": lambda: ema(close.iloc[cut], 20),
            "rsi": lambda: rsi(close.iloc[cut], 14),
            "atr": lambda: atr(high.iloc[cut], low.iloc[cut], close.iloc[cut], 14),
            "obv": lambda: obv(close.iloc[cut], vol.iloc[cut]),
            "vwap": lambda: vwap(high.iloc[cut], low.iloc[cut], close.iloc[cut], vol.iloc[cut]),
        }[name]()
        _assert_same_on_shared_dates(full_series, short)

    bands_full = bollinger(close)
    bands_cut = bollinger(close.iloc[cut])
    for col in ("mid", "upper", "lower"):
        _assert_same_on_shared_dates(bands_full[col], bands_cut[col])

    macd_full = macd(close)
    macd_cut = macd(close.iloc[cut])
    for col in ("macd", "signal", "histogram"):
        _assert_same_on_shared_dates(macd_full[col], macd_cut[col])


def test_the_check_fails_when_a_feature_peeks_ahead():
    hist = _history()
    peeking = feature_frame(hist)
    peeking["rsi_14"] = peeking["rsi_14"].shift(-3)  # deliberately uses prices three days ahead
    with pytest.raises(AssertionError):
        _assert_same_on_shared_dates(peeking["rsi_14"], feature_frame(hist.iloc[:CUT])["rsi_14"])
