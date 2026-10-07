from services.web_search.client import SearchResponse, SearchResult
from services.web_search.summarize import search_summary, summarize_results


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str = "clean summary"):
        self.response_text = response_text
        self.last_prompt = None
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        self.last_prompt = prompt
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("model unavailable")


def _sample_response(query: str = "AAPL news") -> SearchResponse:
    return SearchResponse(
        query=query,
        results=[
            SearchResult(title="Real headline", url="https://a.com", content="Actual news content.", score=0.9),
            SearchResult(title="Ad: buy now", url="https://b.com", content="(Ad) Free stock alerts!", score=0.5),
        ],
        response_time_ms=100,
    )


def test_summarize_results_returns_llm_output():
    llm = _FakeLLM("Apple reported strong iPhone sales.")
    result, sources = summarize_results(_sample_response(), [llm])
    assert result == "Apple reported strong iPhone sales."
    assert llm.calls == 1


def test_summarize_results_prompt_includes_raw_content_and_instructions():
    llm = _FakeLLM()
    summarize_results(_sample_response("TSLA earnings"), [llm], focus="TSLA earnings")
    assert "TSLA earnings" in llm.last_prompt
    assert "Actual news content." in llm.last_prompt
    assert "Deduplicate" in llm.last_prompt


def test_summarize_results_falls_back_to_raw_text_on_llm_failure():
    result, sources = summarize_results(_sample_response(), [_RaisingLLM()])
    assert "Real headline" in result
    assert "Actual news content." in result


def test_summarize_results_falls_through_to_second_llm_before_raw_text():
    healthy = _FakeLLM("second provider answered")
    result, sources = summarize_results(_sample_response(), [_RaisingLLM(), healthy])
    assert result == "second provider answered"
    assert healthy.calls == 1


def test_summarize_results_skips_llm_call_when_no_results():
    empty = SearchResponse(query="nothing found query", results=[], response_time_ms=50)
    llm = _FakeLLM()
    result, sources = summarize_results(empty, [llm])
    assert "nothing found query" in result
    assert llm.calls == 0
    assert sources == []


# --- NFR-5: sources are real, structured, and independent of what the LLM echoes ---

def test_summarize_results_sources_come_from_the_real_results_not_the_llm():
    # The LLM's own output never mentions a URL or the ad headline at all --
    # sources must still list both real results, unaffected by what the
    # summary says or omits.
    llm = _FakeLLM("Apple reported strong iPhone sales.")
    _, sources = summarize_results(_sample_response(), [llm])
    assert sources == [
        {"title": "Real headline", "url": "https://a.com"},
        {"title": "Ad: buy now", "url": "https://b.com"},
    ]


def test_summarize_results_sources_present_even_on_llm_failure():
    _, sources = summarize_results(_sample_response(), [_RaisingLLM()])
    assert len(sources) == 2


def test_search_summary_calls_search_then_summarize(monkeypatch):
    called_with = {}

    def fake_search(query, max_results=5, include_raw_content=False):
        called_with["query"] = query
        called_with["max_results"] = max_results
        return _sample_response(query)

    monkeypatch.setattr("services.web_search.summarize.search", fake_search)
    llm = _FakeLLM("summarized text")

    result, sources = search_summary("AAPL news", [llm], max_results=3)

    assert result == "summarized text"
    assert called_with == {"query": "AAPL news", "max_results": 3}
    assert len(sources) == 2
