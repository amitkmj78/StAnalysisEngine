from services.portfolio_review_service import build_portfolio_review, flag_positions


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
