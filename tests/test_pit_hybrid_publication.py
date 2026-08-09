from datetime import date, timedelta

from services.pit_signal_service import merge_pit_and_live_scores, score_tickers_from_pit


def _rows(ticker, start_price, daily_pct, days=35, start_date=date(2026, 1, 1)):
    rows = []
    price = start_price
    for i in range(days):
        rows.append({"ticker": ticker, "price_date": start_date + timedelta(days=i), "close": price})
        price *= 1 + daily_pct
    return rows


def test_score_tickers_from_pit_returns_all_qualifying_unranked():
    rows = _rows("AAA", 100, 0.01) + _rows("BBB", 100, 0.005) + _rows("CCC", 100, -0.01)
    scored = score_tickers_from_pit(rows, lookback_days=30)
    assert [r["ticker"] for r in scored] == ["AAA", "BBB", "CCC"]
    assert "rank" not in scored[0]


def test_score_tickers_from_pit_excludes_insufficient_history():
    enough = _rows("AAA", 100, 0.01)
    not_enough = _rows("DDD", 100, 0.05, days=10)
    scored = score_tickers_from_pit(enough + not_enough, lookback_days=30)
    tickers = [r["ticker"] for r in scored]
    assert "DDD" not in tickers
    assert "AAA" in tickers


def test_merge_prefers_pit_over_live_for_same_ticker():
    pit_scored = [{"ticker": "AAA", "trailing_return_pct": 10.0}]
    live_scored = [{"ticker": "AAA", "trailing_return_pct": 999.0}]
    ranked = merge_pit_and_live_scores(pit_scored, live_scored, top_n=5)
    assert len(ranked) == 1
    assert ranked[0]["trailing_return_pct"] == 10.0
    assert ranked[0]["data_source"] == "pit"


def test_merge_falls_back_to_live_when_pit_missing_ticker():
    pit_scored = [{"ticker": "AAA", "trailing_return_pct": 10.0}]
    live_scored = [
        {"ticker": "AAA", "trailing_return_pct": 999.0},
        {"ticker": "BBB", "trailing_return_pct": 5.0},
    ]
    ranked = merge_pit_and_live_scores(pit_scored, live_scored, top_n=5)
    by_ticker = {r["ticker"]: r for r in ranked}
    assert by_ticker["AAA"]["data_source"] == "pit"
    assert by_ticker["BBB"]["data_source"] == "live"


def test_merge_ranks_combined_set_and_truncates_to_top_n():
    pit_scored = [{"ticker": "AAA", "trailing_return_pct": 20.0}]
    live_scored = [
        {"ticker": "BBB", "trailing_return_pct": 30.0},
        {"ticker": "CCC", "trailing_return_pct": 10.0},
    ]
    ranked = merge_pit_and_live_scores(pit_scored, live_scored, top_n=2)
    assert [r["ticker"] for r in ranked] == ["BBB", "AAA"]
    assert [r["rank"] for r in ranked] == [1, 2]


def test_merge_with_no_pit_coverage_is_all_live():
    live_scored = [{"ticker": "AAA", "trailing_return_pct": 1.0}]
    ranked = merge_pit_and_live_scores([], live_scored, top_n=5)
    assert ranked[0]["data_source"] == "live"


def test_merge_full_pit_coverage_uses_no_live_fallback():
    pit_scored = [
        {"ticker": "AAA", "trailing_return_pct": 5.0},
        {"ticker": "BBB", "trailing_return_pct": 3.0},
    ]
    ranked = merge_pit_and_live_scores(pit_scored, [], top_n=5)
    assert all(r["data_source"] == "pit" for r in ranked)
