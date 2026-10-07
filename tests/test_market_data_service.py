import pandas as pd

from services import market_data_service


def test_fetch_close_series_always_uses_yahoo_regardless_of_the_admin_price_switch(monkeypatch):
    """Regression: the admin's stock price-source switch (services/
    price_provider.py) was observed live to silently break ^VIX/^VIX3M
    (and intermittently the other internals tickers) once set to Alpaca,
    which has no concept of an index ticker at all. This fetch must
    always go through the Yahoo-only path, never the switchable one."""
    calls = []

    def fake_yahoo_only(ticker, period, auto_adjust=None, interval=None):
        calls.append(ticker)
        idx = pd.bdate_range("2026-01-01", periods=3)
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setattr(market_data_service, "get_cached_history_yahoo_only", fake_yahoo_only)
    result = market_data_service._fetch_close_series("^VIX", "3y")
    assert calls == ["^VIX"]
    assert result is not None


def _fake_breadth():
    idx = pd.bdate_range("2026-01-01", periods=5)
    return pd.DataFrame({"breadth_50dma": [50.0] * 5, "breadth_200dma": [55.0] * 5}, index=idx)


def test_internals_history_is_empty_not_a_keyerror_when_one_aux_ticker_is_missing(monkeypatch):
    # Regression for the Market Regime banner going down: one ticker (e.g.
    # ^VIX) failing a transient Yahoo fetch used to raise a bare KeyError
    # here and take the whole /api/v1/market/regime request down with it.
    idx = pd.bdate_range("2026-01-01", periods=5)
    incomplete_aux = {
        # "^VIX" deliberately missing -- simulates that one ticker's fetch failing.
        "^VIX3M": pd.Series([20.0] * 5, index=idx),
        "XLY": pd.Series([100.0] * 5, index=idx),
        "XLP": pd.Series([80.0] * 5, index=idx),
        "HYG": pd.Series([75.0] * 5, index=idx),
        "IEF": pd.Series([95.0] * 5, index=idx),
        "RSP": pd.Series([150.0] * 5, index=idx),
        "SPY": pd.Series([500.0] * 5, index=idx),
    }
    monkeypatch.setattr(market_data_service, "fetch_sp500_breadth_history", lambda period="3y": _fake_breadth())
    monkeypatch.setattr(market_data_service, "_fetch_closes_parallel", lambda tickers, period: incomplete_aux)

    market_data_service.fetch_market_internals_history.cache.clear()
    result = market_data_service.fetch_market_internals_history("3y")
    assert result.empty


def test_internals_history_builds_normally_when_every_aux_ticker_is_present(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=5)
    full_aux = {
        "^VIX": pd.Series([18.0] * 5, index=idx),
        "^VIX3M": pd.Series([20.0] * 5, index=idx),
        "XLY": pd.Series([100.0] * 5, index=idx),
        "XLP": pd.Series([80.0] * 5, index=idx),
        "HYG": pd.Series([75.0] * 5, index=idx),
        "IEF": pd.Series([95.0] * 5, index=idx),
        "RSP": pd.Series([150.0] * 5, index=idx),
        "SPY": pd.Series([500.0] * 5, index=idx),
    }
    monkeypatch.setattr(market_data_service, "fetch_sp500_breadth_history", lambda period="3y": _fake_breadth())
    monkeypatch.setattr(market_data_service, "_fetch_closes_parallel", lambda tickers, period: full_aux)

    market_data_service.fetch_market_internals_history.cache.clear()
    result = market_data_service.fetch_market_internals_history("3y")
    assert not result.empty
    assert list(result.columns) == ["breadth_50dma", "breadth_200dma", "vix", "vix3m", "xly_xlp", "hyg_ief", "rsp_spy", "spy_close"]
    assert result["vix"].iloc[0] == 18.0


def test_rates_history_is_empty_not_a_keyerror_when_one_ticker_is_missing(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=5)
    # "^MOVE" deliberately missing.
    incomplete = {"^TNX": pd.Series([4.2] * 5, index=idx)}
    monkeypatch.setattr(market_data_service, "_fetch_closes_parallel", lambda tickers, period: incomplete)

    market_data_service.fetch_rates_and_move_history.cache.clear()
    result = market_data_service.fetch_rates_and_move_history("3y")
    assert result.empty
