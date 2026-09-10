"""
Pure helpers for normalizing a position's "acquired_at" value across the
several shapes it can arrive in: a real date/Timestamp carried forward
from an existing row, a pandas NaT/NaN for a position with none, or a
plain missing column. Kept separate from web/backend/routers/portfolio.py
(which uses these) so testing this logic doesn't require importing that
router's whole heavy chain (admin/auth/session config, etc.).
"""

from datetime import date

import numpy as np
import pandas as pd


def is_missing_date(value) -> bool:
    """True for anything that isn't a real, usable date: None, NaN, NaT,
    or something pd.isna chokes on (a plain str/date, which isn't
    missing at all).

    pd.NaT is, perhaps surprisingly, `isinstance(pd.NaT, datetime.date)`
    in this pandas version — so a plain isinstance(value, date) check
    alone would misclassify NaT as a real date. Both Timestamp and date
    values are routed through pd.isna instead, which handles NaT
    correctly and still returns False for anything actually real.
    """
    if value is None:
        return True
    if isinstance(value, (pd.Timestamp, date)):
        return bool(pd.isna(value))
    if isinstance(value, float):
        return bool(np.isnan(value))
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def acquired_at_or_today(value) -> date:
    """
    Normalizes whatever a row's Acquired_At cell happens to hold (a real
    date carried forward from an existing DB row or a CSV's earliest-buy
    date, a pandas NaT/NaN from a position with none, or a plain missing
    column entirely) into a concrete date — today when nothing real is
    available, which is the only honest default for "we don't actually
    know when this was bought."
    """
    if is_missing_date(value):
        return date.today()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value
