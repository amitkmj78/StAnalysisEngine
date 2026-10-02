from unittest.mock import patch

from Agent.newAgent import news_summary, news_summary_with_sources
from services.web_search import SearchResponse, SearchResult


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def invoke(self, prompt: str):
        return _FakeMessage(self.response_text)


def _fake_response():
    return SearchResponse(
        query="AAPL news",
        results=[
            SearchResult(title="Apple beats on earnings", url="https://example.com/a", content="...", score=0.9),
            SearchResult(title="Apple unveils new chip", url="https://example.com/b", content="...", score=0.7),
        ],
        response_time_ms=10,
    )


def test_news_summary_with_sources_returns_real_urls_from_structured_response():
    """ASK-1: sources must come straight from SearchResult objects, never
    reparsed out of the LLM's own summary text."""
    with patch("Agent.newAgent.search", return_value=_fake_response()):
        result = news_summary_with_sources("AAPL", llm=_FakeLLM("Apple had a strong quarter."))

    assert result["summary"] == "Apple had a strong quarter."
    assert result["sources"] == [
        {"title": "Apple beats on earnings", "url": "https://example.com/a"},
        {"title": "Apple unveils new chip", "url": "https://example.com/b"},
    ]


def test_news_summary_unchanged_contract_for_existing_callers():
    """news_summary (used by recommendAgent.py and analysis_service.py)
    must keep returning a plain string, unaffected by the sources plumbing."""
    with patch("Agent.newAgent.search", return_value=_fake_response()):
        result = news_summary("AAPL", llm=_FakeLLM("Apple had a strong quarter."))

    assert result == "Apple had a strong quarter."


def test_no_results_returns_empty_sources_not_an_error():
    empty_response = SearchResponse(query="ZZZZ news", results=[], response_time_ms=5)
    with patch("Agent.newAgent.search", return_value=empty_response):
        result = news_summary_with_sources("ZZZZ", llm=_FakeLLM("should not be called"))

    assert result["sources"] == []
    assert "No recent news found" in result["summary"]
