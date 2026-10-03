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


def test_obv_adds_on_up_days_and_subtracts_on_down_days():
    close = pd.Series([10.0, 11.0, 10.5, 10.5])
    volume = pd.Series([100.0, 200.0, 50.0, 80.0])
    out = obv(close, volume)
    assert list(out) == [0.0, 200.0, 150.0, 150.0]
