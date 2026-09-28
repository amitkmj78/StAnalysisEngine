from unittest.mock import patch

from services.pit_universe_service import capture_universe_membership
from services.stock_finder_service import SP500_UNIVERSE_NAME


@patch("services.pit_universe_service._universe_tickers")
def test_captures_the_real_all_and_sp500_universes(mock_universe_tickers):
    mock_universe_tickers.side_effect = lambda key: {"All": ["AAPL", "MRNA"], SP500_UNIVERSE_NAME: ["AAPL"]}[key]
    rows = capture_universe_membership()
    stock_rows_by_universe = {}
    for row in rows:
        if row["asset_type"] == "stock":
            stock_rows_by_universe.setdefault(row["universe_key"], set()).add(row["ticker"])
    assert stock_rows_by_universe["All"] == {"AAPL", "MRNA"}
    assert stock_rows_by_universe[SP500_UNIVERSE_NAME] == {"AAPL"}


@patch("services.pit_universe_service._universe_tickers")
def test_a_ticker_missing_from_index_map_sample_still_appears_via_all(mock_universe_tickers):
    # Regression test: MRNA has real PIT price history but was never in
    # INDEX_MAP's small SPY/QQQ sample lists, which is exactly why TR-6
    # reconciliation reported it "missing from PIT history" even though
    # Phase 1 had 32 days of its price data on record -- Phase 2 simply
    # never snapshotted its membership. This must not regress: "All" and
    # SP500_UNIVERSE_NAME are captured from the live universe resolver, not
    # the static INDEX_MAP dict, so a real momentum-universe ticker like
    # MRNA is captured even though it isn't hardcoded in INDEX_MAP.
    mock_universe_tickers.side_effect = lambda key: {"All": ["MRNA"], SP500_UNIVERSE_NAME: []}[key]
    rows = capture_universe_membership()
    tickers_captured = {r["ticker"] for r in rows if r["asset_type"] == "stock"}
    assert "MRNA" in tickers_captured


def test_still_captures_index_map_sample_universes_and_funds():
    with patch("services.pit_universe_service._universe_tickers", return_value=[]):
        rows = capture_universe_membership()
    universe_keys = {r["universe_key"] for r in rows if r["asset_type"] == "stock"}
    assert "US - Mega Cap (SPY sample)" in universe_keys
    assert "US - Tech Growth (QQQ sample)" in universe_keys
    assert any(r["asset_type"] == "fund" for r in rows)
