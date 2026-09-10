from datetime import date

import numpy as np
import pandas as pd

from services.acquired_at_utils import acquired_at_or_today, is_missing_date


def test_is_missing_date_true_for_none():
    assert is_missing_date(None) is True


def test_is_missing_date_true_for_nat():
    assert is_missing_date(pd.NaT) is True


def test_is_missing_date_true_for_nan_float():
    assert is_missing_date(float("nan")) is True
    assert is_missing_date(np.nan) is True


def test_is_missing_date_false_for_real_date():
    assert is_missing_date(date(2026, 1, 15)) is False


def test_is_missing_date_false_for_real_timestamp():
    assert is_missing_date(pd.Timestamp("2026-01-15")) is False


def test_acquired_at_or_today_passes_through_real_date():
    d = date(2026, 1, 15)
    assert acquired_at_or_today(d) == d


def test_acquired_at_or_today_converts_timestamp_to_date():
    assert acquired_at_or_today(pd.Timestamp("2026-01-15")) == date(2026, 1, 15)


def test_acquired_at_or_today_defaults_missing_to_today():
    assert acquired_at_or_today(None) == date.today()
    assert acquired_at_or_today(pd.NaT) == date.today()
    assert acquired_at_or_today(float("nan")) == date.today()
