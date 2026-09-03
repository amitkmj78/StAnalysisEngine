from unittest.mock import patch

from services.portfolio_review_service import (
    build_portfolio_review,
    compute_market_values,
    compute_sector_concentration,
    compute_sectors,
    flag_positions,
)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, prompt: str):
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("provider unavailable")


def test_flag_positions_flags_sell_signal():
    positions = [{"ticker": "AAA", "signal": "SELL", "weight_pct": 5.0, "concentrated": False}]
    flagged = flag_positions(positions)
    assert len(flagged) == 1
    assert "quant model's signal is SELL" in flagged[0]["reasons"][0]


def test_flag_positions_flags_concentrated():
    positions = [{"ticker": "BBB", "signal": "HOLD", "weight_pct": 40.0, "concentrated": True}]
    flagged = flag_positions(positions)
    assert len(flagged) == 1
    assert "40%" in flagged[0]["reasons"][0]


def test_flag_positions_flags_bearish_sentiment_with_non_buy_signal():
    positions = [
        {"ticker": "CCC", "signal": "HOLD", "weight_pct": 10.0, "concentrated": False, "sentiment_label": "Bearish"}
    ]
    flagged = flag_positions(positions)
    assert len(flagged) == 1
    assert "Bearish" in flagged[0]["reasons"][0]


def test_flag_positions_does_not_flag_bearish_sentiment_with_buy_signal():
    """A BUY signal despite bearish news is interesting but not the same
    "two independent reads agreeing on caution" case -- only the
    signal+sentiment agreement rules should fire here, and BUY+Bearish
    isn't either of those (not Bearish+non-BUY, not Bullish+BUY)."""
    positions = [
        {"ticker": "DDD", "signal": "BUY", "weight_pct": 10.0, "concentrated": False, "sentiment_label": "Bearish"}
    ]
    flagged = flag_positions(positions)
    assert flagged == []


def test_flag_positions_flags_bullish_agreement():
    positions = [
        {"ticker": "EEE", "signal": "BUY", "weight_pct": 10.0, "concentrated": False, "sentiment_label": "Bullish"}
    ]
    flagged = flag_positions(positions)
    assert len(flagged) == 1
    assert "both read Bullish" in flagged[0]["reasons"][0]


def test_flag_positions_leaves_unremarkable_positions_unflagged():
    positions = [
        {"ticker": "FFF", "signal": "HOLD", "weight_pct": 5.0, "concentrated": False, "sentiment_label": "Neutral"}
    ]
    assert flag_positions(positions) == []


def test_flag_positions_can_flag_for_multiple_reasons():
    positions = [
        {
            "ticker": "GGG",
            "signal": "SELL",
            "weight_pct": 30.0,
            "concentrated": True,
            "sentiment_label": "Bearish",
        }
    ]
    flagged = flag_positions(positions)
    assert len(flagged) == 1
    # SELL signal, concentrated, and Bearish+non-BUY are independent checks --
    # all three legitimately apply here at once.
    assert len(flagged[0]["reasons"]) == 3


def test_compute_market_values_multiplies_shares_by_live_price():
    positions = [{"ticker": "AAA", "shares": 10.0}, {"ticker": "BBB", "shares": 2.0}]
    with patch(
        "services.portfolio_review_service.get_effective_price",
        side_effect=lambda t: {"AAA": 100.0, "BBB": 250.0}[t],
    ):
        result = compute_market_values(positions)
    assert result == {"AAA": 1000.0, "BBB": 500.0}


def test_compute_market_values_omits_tickers_with_no_price():
    positions = [{"ticker": "AAA", "shares": 10.0}, {"ticker": "ZZZ", "shares": 5.0}]
    with patch(
        "services.portfolio_review_service.get_effective_price",
        side_effect=lambda t: 100.0 if t == "AAA" else None,
    ):
        result = compute_market_values(positions)
    assert result == {"AAA": 1000.0}
    assert "ZZZ" not in result


def test_compute_sectors_omits_tickers_with_no_sector_info():
    with patch(
        "services.portfolio_review_service.get_cached_info",
        side_effect=lambda t: {"sector": "Technology"} if t == "AAPL" else {},
    ):
        result = compute_sectors(["AAPL", "SOME_FUND"])
    assert result == {"AAPL": "Technology"}


def test_compute_sector_concentration_flags_sector_above_threshold():
    positions = [
        {"ticker": "XLK", "sector": "Technology", "market_value": 4000.0},
        {"ticker": "META", "sector": "Technology", "market_value": 3000.0},
        {"ticker": "PG", "sector": "Consumer Defensive", "market_value": 3000.0},
    ]
    # Technology = 7000 / 10000 = 70% >= 40% threshold
    result = compute_sector_concentration(positions)
    assert "Technology" in result
    assert result["Technology"]["weight_pct"] == 70.0
    assert result["Technology"]["market_value"] == 7000.0
    assert set(result["Technology"]["tickers"]) == {"XLK", "META"}
    assert "Consumer Defensive" not in result


def test_compute_sector_concentration_ignores_positions_with_no_sector():
    positions = [
        {"ticker": "XLK", "sector": None, "market_value": 9000.0},
        {"ticker": "PG", "sector": "Consumer Defensive", "market_value": 1000.0},
    ]
    assert compute_sector_concentration(positions) == {}


def test_flag_positions_flags_sector_concentration_even_without_single_position_flag():
    """A position well under the 25% single-position threshold should
    still get flagged if its sector collectively dominates the
    portfolio -- this is the whole point of the sector check."""
    positions = [
        {"ticker": "XLK", "signal": "HOLD", "weight_pct": 20.0, "concentrated": False,
         "sector": "Technology", "market_value": 4000.0},
        {"ticker": "META", "signal": "HOLD", "weight_pct": 25.0, "concentrated": False,
         "sector": "Technology", "market_value": 5000.0},
        {"ticker": "PG", "signal": "HOLD", "weight_pct": 10.0, "concentrated": False,
         "sector": "Consumer Defensive", "market_value": 1000.0},
    ]
    flagged = flag_positions(positions)
    flagged_tickers = {f["ticker"] for f in flagged}
    assert flagged_tickers == {"XLK", "META"}
    assert any("Technology sector concentration" in r for r in next(f for f in flagged if f["ticker"] == "XLK")["reasons"])


def test_build_portfolio_review_returns_none_for_empty_flagged_list():
    assert build_portfolio_review([_FakeLLM("should not be called")], []) is None


def test_build_portfolio_review_returns_llm_text():
    flagged = [{"ticker": "AAA", "signal": "SELL", "expected_return_pct": -6.0, "weight_pct": 12.0,
                "sentiment_label": "Bearish", "reasons": ["the quant model's signal is SELL"]}]
    llm = _FakeLLM("AAA stands out with a SELL signal and bearish sentiment reinforcing it.")
    result = build_portfolio_review([llm], flagged)
    assert result == "AAA stands out with a SELL signal and bearish sentiment reinforcing it."


def test_build_portfolio_review_returns_none_when_all_providers_fail():
    flagged = [{"ticker": "AAA", "signal": "SELL", "expected_return_pct": -6.0, "weight_pct": 12.0,
                "sentiment_label": "Bearish", "reasons": ["the quant model's signal is SELL"]}]
    assert build_portfolio_review([_RaisingLLM()], flagged) is None
