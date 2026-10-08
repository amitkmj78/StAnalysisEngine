from unittest.mock import patch

from services import market_overview_service as mod


def test_get_market_overview_returns_labeled_price_and_change():
    mod.get_market_overview.cache.clear()

    def fake_fast_info(ticker):
        return {"^GSPC": {"lastPrice": 5100.0, "previousClose": 5000.0}}.get(ticker, {})

    with patch.object(mod.yf, "Ticker") as mock_ticker:
        mock_ticker.side_effect = lambda t: type("T", (), {"fast_info": fake_fast_info(t)})()
        result = mod.get_market_overview()

    gspc = next(r for r in result if r["ticker"] == "^GSPC")
    assert gspc["label"] == "S&P 500"
    assert gspc["price"] == 5100.0
    assert gspc["change_pct"] == 2.0


def test_get_market_overview_handles_a_failed_fetch_gracefully():
    mod.get_market_overview.cache.clear()

    with patch.object(mod.yf, "Ticker", side_effect=Exception("rate limited")):
        result = mod.get_market_overview()

    assert len(result) == len(mod.MARKET_OVERVIEW_TICKERS)
    assert all(r["price"] is None and r["change_pct"] is None for r in result)


def test_get_market_overview_returns_all_three_labeled_indices():
    mod.get_market_overview.cache.clear()

    with patch.object(mod.yf, "Ticker", side_effect=Exception("skip")):
        result = mod.get_market_overview()

    labels = {r["label"] for r in result}
    assert labels == {"S&P 500", "Nasdaq", "Dow"}
