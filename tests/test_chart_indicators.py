import numpy as np
import pandas as pd
import pytest

from services.chart_indicators import atr, bollinger, ema, macd, obv, rsi, sma, vwap


def test_sma_matches_hand_mean():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_matches_manual_recursion():
    s = pd.Series([10.0, 11.0, 12.0])
    alpha = 2 / (3 + 1)
    expected = [10.0]
    for x in [11.0, 12.0]:
        expected.append(alpha * x + (1 - alpha) * expected[-1])
    assert list(ema(s, 3).round(10)) == pytest.approx(expected, abs=1e-9)


def test_rsi_is_100_on_rising_and_0_on_falling_history():
    up = pd.Series(np.arange(1.0, 40.0))
    down = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_is_near_50_when_gains_and_losses_balance():
    alternating = pd.Series([100.0 + (i % 2) for i in range(200)])
    assert 45 < rsi(alternating, 14).iloc[-1] < 55


def test_macd_is_zero_on_constant_price():
    m = macd(pd.Series([50.0] * 60))
    assert m["macd"].iloc[-1] == pytest.approx(0.0)
    assert m["histogram"].iloc[-1] == pytest.approx(0.0)


def test_bollinger_bands_collapse_on_constant_price_and_widen_on_noise():
    flat = bollinger(pd.Series([20.0] * 25))
    assert flat["upper"].iloc[-1] == pytest.approx(20.0)
    noisy = bollinger(pd.Series([10.0, 12.0] * 15))
    assert noisy["upper"].iloc[-1] - noisy["lower"].iloc[-1] == pytest.approx(4.0 * 1.0, abs=1e-9)


def test_vwap_matches_hand_calculation_for_two_bars():
    high = pd.Series([12.0, 14.0])
    low = pd.Series([10.0, 12.0])
    close = pd.Series([11.0, 13.0])
    volume = pd.Series([100.0, 300.0])
    out = vwap(high, low, close, volume)
    tp1, tp2 = (12 + 10 + 11) / 3, (14 + 12 + 13) / 3
    assert out.iloc[0] == pytest.approx(tp1)
    assert out.iloc[1] == pytest.approx((tp1 * 100 + tp2 * 300) / 400)


def test_atr_is_true_range_mean_on_constant_bars():
    high = pd.Series([11.0] * 20)
    low = pd.Series([9.0] * 20)
    close = pd.Series([10.0] * 20)
    assert atr(high, low, close, 14).iloc[-1] == pytest.approx(2.0)


def test_atr_is_wilder_smoothed_from_a_simple_mean_seed():
    # True ranges by hand: 2, 2, 2, 5, 2. With n = 3 the seed is mean(2, 2, 2) = 2,
    # then atr = (prev * 2 + TR) / 3: 3 at index 3, and 8/3 at index 4.
    high = pd.Series([11.0, 12.0, 13.0, 17.0, 15.0])
    low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0])
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    out = atr(high, low, close, 3)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(8.0 / 3.0)


def test_obv_adds_on_up_days_and_subtracts_on_down_days():
    close = pd.Series([10.0, 11.0, 10.5, 10.5])
    volume = pd.Series([100.0, 200.0, 50.0, 80.0])
    out = obv(close, volume)
    assert list(out) == [0.0, 200.0, 150.0, 150.0]


def test_indicators_match_reference_library_on_a_fixed_series():
    # CHT-3 acceptance: values match a reference calculation within 0.1%.
    ta = pytest.importorskip("ta")
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    high = close * (1 + rng.uniform(0, 0.01, 300))
    low = close * (1 - rng.uniform(0, 0.01, 300))
    volume = pd.Series(rng.integers(1_000, 10_000, 300).astype(float))

    def assert_close(ours, ref, skip):
        a = ours.iloc[skip:].to_numpy(dtype=float)
        b = pd.Series(ref).iloc[skip:].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-9, equal_nan=True)

    assert_close(rsi(close, 14), ta.momentum.RSIIndicator(close, 14).rsi(), 30)
    assert_close(atr(high, low, close, 14), ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range(), 20)
    assert_close(macd(close)["macd"], ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9).macd(), 40)
    assert_close(sma(close, 20), ta.trend.SMAIndicator(close, 20).sma_indicator(), 25)
    assert_close(ema(close, 20), ta.trend.EMAIndicator(close, 20).ema_indicator(), 25)
    assert_close(bollinger(close)["upper"], ta.volatility.BollingerBands(close, 20, 2).bollinger_hband(), 25)
