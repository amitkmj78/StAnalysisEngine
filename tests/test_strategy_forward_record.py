from services.strategy_forward_record import MIN_DAYS_SINCE_PUBLISH, _needs_sp500, _resolve_tickers

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM"]


def test_resolve_tickers_hand_picked_uses_definitions_own_list():
    definition = {"source": "hand_picked", "tickers": ["aapl", "MSFT", "aapl"]}
    assert _resolve_tickers(definition, UNIVERSE) == ["AAPL", "MSFT"]


def test_resolve_tickers_portfolio_weights_uses_weight_keys_not_source():
    # weights takes priority over `source`, mirroring strategy_builder.py::backtest's own branch order.
    definition = {"source": "hand_picked", "tickers": [], "weights": {"nvda": 0.6, "TSLA": 0.4}}
    assert _resolve_tickers(definition, UNIVERSE) == ["NVDA", "TSLA"]


def test_resolve_tickers_random_sample_is_deterministic_given_the_same_seed():
    definition = {"source": "random_sample", "sample_seed": 42, "sample_size": 3}
    first = _resolve_tickers(definition, UNIVERSE)
    second = _resolve_tickers(definition, UNIVERSE)
    assert first == second
    assert len(first) == 3
    assert set(first).issubset(set(UNIVERSE))


def test_resolve_tickers_random_sample_without_a_recorded_seed_fails_open_to_empty():
    """Can't reproduce a draw with no seed -- never guesses a ticker list."""
    definition = {"source": "random_sample", "sample_seed": None}
    assert _resolve_tickers(definition, UNIVERSE) == []


def test_resolve_tickers_random_sample_falls_back_to_tickers_length_for_size():
    # No sample_size recorded -- falls back to max(len(tickers), 5), the
    # same floor strategy_builder.py::backtest's own branch uses.
    definition = {"source": "random_sample", "sample_seed": 7, "tickers": ["a", "b", "c", "d", "e", "f", "g"]}
    result = _resolve_tickers(definition, UNIVERSE)
    assert len(result) == 7


def test_needs_sp500_only_for_random_sample_without_weights():
    assert _needs_sp500({"source": "random_sample"}) is True
    assert _needs_sp500({"source": "hand_picked"}) is False
    assert _needs_sp500({"source": "random_sample", "weights": {"AAPL": 1.0}}) is False


def test_min_days_since_publish_is_a_real_positive_gate():
    assert MIN_DAYS_SINCE_PUBLISH >= 1
