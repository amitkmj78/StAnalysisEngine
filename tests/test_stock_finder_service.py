"""
_annualized_return regression tests -- a real, reported bug: annualizing a
young ticker's short real price history (e.g. a 2025 spin-off with only a
few months of trading) extrapolates a modest short-term move into an
absurd "3-year annualized return" (a specific case: 1,089% on a ticker
with well under 3 years of real history), which then won a "Long Term"
ranking outright. No live network -- every test builds its own synthetic
price series.
"""

import pandas as pd
import pytest

from services.stock_finder_service import _annualized_return, _pct_return


def _prices(values, start="2020-01-01", freq="B"):
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


def test_annualized_return_none_when_history_too_short():
    # ~4 months of real history (well under min_years=2.9) -- even though
    # the series has 100 rows and a real (if big) move, it must not be
    # annualized, exactly the SNDK-style bug.
    close = _prices([100.0] * 20 + list(range(100, 200)))
    assert _annualized_return(close) is None


def test_annualized_return_computes_normally_with_full_history():
    # ~3 years of trading days, doubling smoothly -- a genuinely reliable
    # annualized figure should come back, not None. Doubling over 3 years
    # is 2**(1/3)-1 ≈ 26% annualized, not 100%.
    days = 756
    values = [100 * (2 ** (i / days)) for i in range(days)]
    close = _prices(values)
    result = _annualized_return(close)
    assert result is not None
    assert result == pytest.approx(25.99, abs=1.0)


def test_annualized_return_sanity_bound_rejects_extreme_value():
    # Full 3 years of history, but an absurd 50x move over that span --
    # annualizes to a number far past the 200% sanity bound, so it must
    # come back None (flagged for review) rather than a trusted #1-ranking
    # figure.
    days = 756
    close = _prices([100.0] * (days - 1) + [5000.0])
    result = _annualized_return(close)
    assert result is None


def test_annualized_return_rejects_extreme_negative_too():
    # 1000 -> 0.001 over 3 years annualizes to about -99% (well past the
    # -95% floor) -- a near-total wipeout that's still a data/edge-case
    # concern, not just the positive-extreme direction.
    days = 756
    close = _prices([1000.0] * (days - 1) + [0.001])
    result = _annualized_return(close)
    assert result is None


def test_annualized_return_none_for_empty_or_single_point():
    assert _annualized_return(pd.Series(dtype=float)) is None
    assert _annualized_return(_prices([100.0])) is None


def test_pct_return_unchanged_none_when_lookback_exceeds_history():
    # Existing "not enough data" convention this fix mirrors -- confirms it
    # still works as before.
    close = _prices([100.0, 101.0, 102.0])
    assert _pct_return(close, 10) is None
