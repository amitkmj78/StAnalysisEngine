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

from services.stock_finder_service import _annualized_return, _pct_return, _build_stock_row, rank_stocks_by_window_return
import services.stock_finder_service as sfs


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


# ---------------------------------------------------------------------------
# rank_stocks_by_window_return -- shared by /momentum/top-performers and the
# /portfolio/compare endpoint's top_stocks, so both surfaces rank identically.
# ---------------------------------------------------------------------------


def _synthetic_universe_df():
    return pd.DataFrame([
        {"Ticker": "AAA", "Name": "Alpha Co", "Sector": "Technology", "Return 90D %": 40.0, "1Y Return %": 10.0},
        {"Ticker": "BBB", "Name": "Beta Co", "Sector": "Energy", "Return 90D %": 20.0, "1Y Return %": 90.0},
        {"Ticker": "CCC", "Name": "Gamma Co", "Sector": "Technology", "Return 90D %": None, "1Y Return %": 5.0},
    ])


def test_rank_stocks_by_window_return_uses_correct_column(monkeypatch):
    monkeypatch.setattr(sfs, "get_stock_finder_table", lambda universe_key: _synthetic_universe_df())

    ranked_90d = rank_stocks_by_window_return("90D", "All", 10)
    assert [r["ticker"] for r in ranked_90d] == ["AAA", "BBB"]  # CCC dropped (NaN on this column)

    ranked_1y = rank_stocks_by_window_return("1Y", "All", 10)
    assert [r["ticker"] for r in ranked_1y] == ["BBB", "AAA", "CCC"]  # sorted by 1Y Return %, not 90D


def test_rank_stocks_by_window_return_owned_flag(monkeypatch):
    monkeypatch.setattr(sfs, "get_stock_finder_table", lambda universe_key: _synthetic_universe_df())
    ranked = rank_stocks_by_window_return("90D", "All", 10, owned_tickers={"AAA"})
    owned = {r["ticker"]: r["owned"] for r in ranked}
    assert owned["AAA"] is True
    assert owned["BBB"] is False


def test_rank_stocks_by_window_return_gics_sector_renamed(monkeypatch):
    monkeypatch.setattr(sfs, "get_stock_finder_table", lambda universe_key: _synthetic_universe_df())
    ranked = rank_stocks_by_window_return("90D", "All", 10)
    assert ranked[0]["sector"] == "Information Technology"  # Technology -> GICS renamed


def test_rank_stocks_by_window_return_unknown_window_raises(monkeypatch):
    monkeypatch.setattr(sfs, "get_stock_finder_table", lambda universe_key: _synthetic_universe_df())
    with pytest.raises(ValueError):
        rank_stocks_by_window_return("7D", "All", 10)


def test_rank_stocks_by_window_return_empty_universe(monkeypatch):
    monkeypatch.setattr(sfs, "get_stock_finder_table", lambda universe_key: pd.DataFrame())
    assert rank_stocks_by_window_return("90D", "All", 10) == []


# ---------------------------------------------------------------------------
# _build_stock_row -- SCN-1/SCN-2: dividend yield and the sparkline column.
# No live network -- get_cached_history/get_cached_info are monkeypatched.
# ---------------------------------------------------------------------------


def _synthetic_info(**overrides):
    info = {
        "shortName": "Alpha Co",
        "sector": "Technology",
        "industry": "Software",
        "marketCap": 1_000_000_000,
        "forwardPE": 20.0,
        "dividendYield": 0.021,
        "revenueGrowth": 0.05,
        "earningsGrowth": 0.05,
    }
    info.update(overrides)
    return info


def _synthetic_hist(n=100):
    close = _prices(list(range(100, 100 + n)))
    volume = pd.Series([1_000_000.0] * n, index=close.index)
    return pd.DataFrame({"Close": close, "Volume": volume})


def test_build_stock_row_spark_90d_is_last_90_closes(monkeypatch):
    monkeypatch.setattr(sfs, "get_cached_history", lambda ticker, period, auto_adjust=True: _synthetic_hist(100))
    monkeypatch.setattr(sfs, "get_cached_info", lambda ticker: _synthetic_info())

    row = _build_stock_row("AAA")

    assert row is not None
    assert row["Spark 90D"] == pytest.approx(list(range(110, 200)), abs=0.01)
    assert len(row["Spark 90D"]) == 90


def test_build_stock_row_spark_90d_shorter_than_90_uses_full_history(monkeypatch):
    monkeypatch.setattr(sfs, "get_cached_history", lambda ticker, period, auto_adjust=True: _synthetic_hist(70))
    monkeypatch.setattr(sfs, "get_cached_info", lambda ticker: _synthetic_info())

    row = _build_stock_row("AAA")

    assert row is not None
    assert len(row["Spark 90D"]) == 70


def test_build_stock_row_dividend_yield_passed_through_unscaled(monkeypatch):
    # Unlike revenueGrowth/earningsGrowth, yfinance's dividendYield is
    # already a plain percent (confirmed live: MSFT's raw value is 0.77,
    # meaning 0.77%) -- must NOT be run through _safe_percent's
    # fraction-detection heuristic, which would 100x a real sub-1% yield.
    monkeypatch.setattr(sfs, "get_cached_history", lambda ticker, period, auto_adjust=True: _synthetic_hist(100))

    monkeypatch.setattr(sfs, "get_cached_info", lambda ticker: _synthetic_info(dividendYield=0.77))
    assert _build_stock_row("AAA")["Dividend Yield %"] == pytest.approx(0.77)

    monkeypatch.setattr(sfs, "get_cached_info", lambda ticker: _synthetic_info(dividendYield=2.43))
    assert _build_stock_row("AAA")["Dividend Yield %"] == pytest.approx(2.43)


def test_build_stock_row_dividend_yield_none_when_missing(monkeypatch):
    monkeypatch.setattr(sfs, "get_cached_history", lambda ticker, period, auto_adjust=True: _synthetic_hist(100))
    monkeypatch.setattr(sfs, "get_cached_info", lambda ticker: _synthetic_info(dividendYield=None))

    assert _build_stock_row("AAA")["Dividend Yield %"] is None
