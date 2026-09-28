from datetime import date

import numpy as np
import pandas as pd

from services.stock_detail_service import next_earnings_date, recent_dividends, select_peers


def _universe():
    return pd.DataFrame(
        [
            {"Ticker": "AAPL", "Name": "Apple", "GICS Sector": "Information Technology", "Market Cap ($B)": 3000.0},
            {"Ticker": "MSFT", "Name": "Microsoft", "GICS Sector": "Information Technology", "Market Cap ($B)": 2900.0},
            {"Ticker": "NVDA", "Name": "Nvidia", "GICS Sector": "Information Technology", "Market Cap ($B)": 3200.0},
            {"Ticker": "ORCL", "Name": "Oracle", "GICS Sector": "Information Technology", "Market Cap ($B)": 500.0},
            {"Ticker": "CRM", "Name": "Salesforce", "GICS Sector": "Information Technology", "Market Cap ($B)": 250.0},
            {"Ticker": "ADBE", "Name": "Adobe", "GICS Sector": "Information Technology", "Market Cap ($B)": 200.0},
            {"Ticker": "JNJ", "Name": "Johnson & Johnson", "GICS Sector": "Health Care", "Market Cap ($B)": 400.0},
        ]
    )


def test_select_peers_prefers_closest_market_cap_within_same_sector():
    peers = select_peers("AAPL", _universe(), top_n=3)
    tickers = [p["ticker"] for p in peers]
    # AAPL=3000: closest by |cap - 3000| among same-sector, non-self tickers
    # is MSFT(2900, dist 100), NVDA(3200, dist 200), ORCL(500, dist 2500).
    assert tickers == ["MSFT", "NVDA", "ORCL"]


def test_select_peers_excludes_other_sectors():
    peers = select_peers("AAPL", _universe(), top_n=10)
    assert "JNJ" not in [p["ticker"] for p in peers]


def test_select_peers_unknown_ticker_returns_empty():
    assert select_peers("ZZZZ", _universe()) == []


def test_select_peers_missing_market_cap_returns_empty():
    df = pd.DataFrame([{"Ticker": "XYZ", "Name": "X", "GICS Sector": "Energy", "Market Cap ($B)": np.nan}])
    assert select_peers("XYZ", df) == []


def test_next_earnings_date_picks_the_soonest_upcoming_row():
    earnings = pd.DataFrame(
        {"EPS Estimate": [1.5, 1.8, None]},
        index=pd.to_datetime(["2026-01-01", "2026-03-15", "2026-06-15"]),
    )
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": 1.8}


def test_next_earnings_date_none_when_no_future_row():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-01-01"]))
    assert next_earnings_date(earnings, as_of=date(2026, 2, 1)) is None


def test_next_earnings_date_empty_input():
    assert next_earnings_date(pd.DataFrame()) is None


def test_next_earnings_date_handles_tz_aware_index_without_crashing():
    earnings = pd.DataFrame(
        {"EPS Estimate": [2.0]},
        index=pd.date_range("2026-03-15", periods=1, tz="America/New_York"),
    )
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": 2.0}


def test_next_earnings_date_none_eps_estimate_when_not_provided():
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.to_datetime(["2026-03-15"]))
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": None}


def test_recent_dividends_most_recent_first():
    dividends = pd.Series([0.20, 0.22, 0.24], index=pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01"]))
    result = recent_dividends(dividends, top_n=2)
    assert result == [{"date": "2026-07-01", "amount": 0.24}, {"date": "2026-04-01", "amount": 0.22}]


def test_recent_dividends_empty_series():
    assert recent_dividends(pd.Series(dtype=float)) == []
