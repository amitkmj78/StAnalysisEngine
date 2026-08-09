from services.ranking_utils import compute_position_concentration, rank_tickers_against_universe


def test_rank_orders_by_trailing_return_descending():
    universe = [
        {"ticker": "AAA", "trailing_return_pct": 5.0},
        {"ticker": "BBB", "trailing_return_pct": 20.0},
        {"ticker": "CCC", "trailing_return_pct": -3.0},
    ]
    result = rank_tickers_against_universe(universe, ["AAA", "BBB", "CCC"])
    assert result["BBB"]["rank"] == 1
    assert result["AAA"]["rank"] == 2
    assert result["CCC"]["rank"] == 3
    assert result["AAA"]["universe_size"] == 3
    assert result["AAA"]["trailing_return_pct"] == 5.0


def test_rank_missing_ticker_gets_none_not_a_guess():
    universe = [{"ticker": "AAA", "trailing_return_pct": 5.0}]
    result = rank_tickers_against_universe(universe, ["AAA", "ZZZ"])
    assert result["ZZZ"]["rank"] is None
    assert result["ZZZ"]["trailing_return_pct"] is None
    assert result["ZZZ"]["universe_size"] == 1


def test_rank_excludes_rows_with_no_trailing_return_from_ranking():
    universe = [
        {"ticker": "AAA", "trailing_return_pct": 5.0},
        {"ticker": "BBB", "trailing_return_pct": None},
    ]
    result = rank_tickers_against_universe(universe, ["AAA", "BBB"])
    assert result["AAA"]["rank"] == 1
    assert result["AAA"]["universe_size"] == 1
    assert result["BBB"]["rank"] is None


def test_rank_empty_universe_returns_none_for_everyone():
    result = rank_tickers_against_universe([], ["AAA"])
    assert result["AAA"] == {"rank": None, "universe_size": 0, "trailing_return_pct": None}


def test_concentration_flags_position_at_or_above_threshold():
    positions = [
        {"ticker": "AAA", "market_value": 8000.0},
        {"ticker": "BBB", "market_value": 2000.0},
    ]
    result = compute_position_concentration(positions, threshold_pct=25.0)
    by_ticker = {r["ticker"]: r for r in result}
    assert by_ticker["AAA"]["weight_pct"] == 80.0
    assert by_ticker["AAA"]["concentrated"] is True
    assert by_ticker["BBB"]["weight_pct"] == 20.0
    assert by_ticker["BBB"]["concentrated"] is False


def test_concentration_exactly_at_threshold_counts_as_concentrated():
    positions = [
        {"ticker": "AAA", "market_value": 25.0},
        {"ticker": "BBB", "market_value": 75.0},
    ]
    result = compute_position_concentration(positions, threshold_pct=25.0)
    by_ticker = {r["ticker"]: r for r in result}
    assert by_ticker["AAA"]["concentrated"] is True


def test_concentration_zero_total_value_is_safe_not_a_divide_by_zero():
    positions = [{"ticker": "AAA", "market_value": 0.0}, {"ticker": "BBB", "market_value": 0.0}]
    result = compute_position_concentration(positions)
    assert all(r["weight_pct"] == 0.0 and r["concentrated"] is False for r in result)


def test_concentration_single_position_is_fully_concentrated():
    positions = [{"ticker": "AAA", "market_value": 500.0}]
    result = compute_position_concentration(positions, threshold_pct=25.0)
    assert result[0]["weight_pct"] == 100.0
    assert result[0]["concentrated"] is True
