from unittest.mock import patch

from services.sentiment_service import score_ticker_sentiment, score_tickers_sentiment


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("provider unavailable")


def test_score_ticker_sentiment_parses_bullish_label_and_reasoning():
    llm = _FakeLLM("SENTIMENT: Bullish\nREASON: Strong earnings beat with raised guidance.")
    with patch("services.sentiment_service.get_sentiment_summary", return_value="fake context"):
        result = score_ticker_sentiment("AAPL", [llm])
    assert result == {"label": "Bullish", "reasoning": "Strong earnings beat with raised guidance."}


def test_score_ticker_sentiment_is_case_and_punctuation_tolerant():
    llm = _FakeLLM("sentiment: bearish.\nreason: Guidance cut spooked investors.")
    with patch("services.sentiment_service.get_sentiment_summary", return_value="fake context"):
        result = score_ticker_sentiment("XYZ", [llm])
    assert result == {"label": "Bearish", "reasoning": "Guidance cut spooked investors."}


def test_score_ticker_sentiment_returns_none_on_unparseable_response():
    llm = _FakeLLM("I'm not sure how to answer that.")
    with patch("services.sentiment_service.get_sentiment_summary", return_value="fake context"):
        result = score_ticker_sentiment("AAPL", [llm])
    assert result == {"label": None, "reasoning": None}


def test_score_ticker_sentiment_returns_none_when_all_providers_fail():
    with patch("services.sentiment_service.get_sentiment_summary", return_value="fake context"):
        result = score_ticker_sentiment("AAPL", [_RaisingLLM()])
    assert result == {"label": None, "reasoning": None}


def test_score_ticker_sentiment_returns_none_when_summary_lookup_raises():
    with patch("services.sentiment_service.get_sentiment_summary", side_effect=RuntimeError("search down")):
        result = score_ticker_sentiment("AAPL", [_FakeLLM("SENTIMENT: Bullish\nREASON: n/a")])
    assert result == {"label": None, "reasoning": None}


def test_score_tickers_sentiment_returns_one_entry_per_input_ticker():
    llm = _FakeLLM("SENTIMENT: Neutral\nREASON: Mixed signals in recent coverage.")
    with patch("services.sentiment_service.get_sentiment_summary", return_value="fake context"):
        results = score_tickers_sentiment(["AAPL", "MSFT"], [llm])
    assert set(results.keys()) == {"AAPL", "MSFT"}
    assert all(r["label"] == "Neutral" for r in results.values())


def test_score_tickers_sentiment_empty_input_returns_empty_dict():
    assert score_tickers_sentiment([], [_FakeLLM("SENTIMENT: Bullish\nREASON: n/a")]) == {}
