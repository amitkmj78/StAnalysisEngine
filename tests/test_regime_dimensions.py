import numpy as np
import pandas as pd

from services.regime_dimensions import (
    credit_reading,
    divergence_flag,
    rates_reading,
    risk_appetite_reading,
    breadth_reading,
)


def _frame(n=300, **cols):
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    data = {}
    for name, value in cols.items():
        data[name] = np.full(n, value, dtype=float) if np.isscalar(value) else np.asarray(value, dtype=float)
    return pd.DataFrame(data, index=idx)


def _yield_path(start, end, n=300):
    """Flat until the last three months, then a straight move over those 63 sessions."""
    flat = np.full(n - 63, start)
    return np.concatenate([flat, np.linspace(start, end, 63)])


# REG-4 rates

def test_rates_minus_one_when_yield_rose_more_than_half_a_point():
    df = _frame(tnx=_yield_path(4.0, 4.6), move=80.0)
    assert rates_reading(df)["score"] == -1


def test_rates_exactly_half_a_point_is_not_a_rise_signal():
    df = _frame(tnx=_yield_path(4.0, 4.5), move=80.0)
    r = rates_reading(df)
    assert r["yield_change_pts"] == 0.5 and r["score"] == 0


def test_rates_minus_one_when_move_above_120_even_if_yield_flat():
    df = _frame(tnx=_yield_path(4.0, 4.0), move=121.0)
    assert rates_reading(df)["score"] == -1


def test_rates_plus_one_when_yields_fell_and_move_calm():
    df = _frame(tnx=_yield_path(4.5, 4.2), move=85.0)
    assert rates_reading(df)["score"] == 1


def test_rates_zero_when_yields_fell_but_move_not_calm_enough():
    df = _frame(tnx=_yield_path(4.5, 4.2), move=95.0)
    assert rates_reading(df)["score"] == 0


def test_rates_unavailable_without_yield_history_is_none_not_guessed():
    df = _frame(move=80.0)
    assert rates_reading(df)["score"] is None


# REG-5 credit

def test_credit_plus_one_when_ratio_above_50_day_average():
    values = np.concatenate([np.full(249, 1.0), np.full(50, 1.0), np.full(1, 1.05)])
    assert credit_reading(_frame(hyg_ief=values))["score"] == 1


def test_credit_minus_one_when_ratio_below_average():
    values = np.concatenate([np.full(249, 1.05), np.full(50, 1.05), np.full(1, 1.0)])
    assert credit_reading(_frame(hyg_ief=values))["score"] == -1


def test_credit_needs_a_full_50_day_window():
    assert credit_reading(_frame(n=30, hyg_ief=1.0))["score"] is None


# REG-6 breadth

def test_breadth_reports_both_windows_for_spy_and_stocks():
    df = _frame(spy_close=np.linspace(100, 120, 300), breadth_50dma=55.0, breadth_200dma=48.0)
    r = breadth_reading(df)
    assert r["spy_above_50dma"] is True and r["spy_above_200dma"] is True
    assert r["pct_above_50dma"] == 55.0 and r["pct_above_200dma"] == 48.0
    assert "50-day" in r["text"] and "200-day" in r["text"]


# REG-7 divergence

def test_divergence_flags_spy_up_with_narrow_breadth():
    df = _frame(spy_close=np.linspace(100, 120, 300), breadth_50dma=34.9)
    assert divergence_flag(df)["flag"] is True


def test_divergence_not_flagged_at_exactly_35_percent():
    df = _frame(spy_close=np.linspace(100, 120, 300), breadth_50dma=35.0)
    assert divergence_flag(df)["flag"] is False


def test_divergence_not_flagged_when_spy_below_its_average():
    df = _frame(spy_close=np.linspace(120, 100, 300), breadth_50dma=20.0)
    assert divergence_flag(df)["flag"] is False


# REG-8 risk appetite

def test_risk_appetite_names_ratio_and_window_in_text():
    df = _frame(rsp_spy=np.linspace(1.0, 0.98, 300))
    r = risk_appetite_reading(df)
    assert "Equal-weight vs S&P 500" in r["text"] and "last 3 months" in r["text"]
    assert r["change_pct"] < 0
