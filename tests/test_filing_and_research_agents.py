from Agent.filingAgent import UNVERIFIED_DISCLOSURE as FILING_DISCLOSURE
from Agent.filingAgent import filings_analysis
from Agent.reasearchAgent import UNVERIFIED_DISCLOSURE as RESEARCH_DISCLOSURE
from Agent.reasearchAgent import research


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def invoke(self, prompt: str):
        return _FakeMessage("Some analysis text.")


# --- NFR-5: neither tool fetches real data, so every real response must
# carry its own unverified disclosure rather than reading as fact-checked.

def test_filings_analysis_appends_unverified_disclosure():
    result = filings_analysis("AAPL", llm=_FakeLLM())
    assert "Some analysis text." in result
    assert FILING_DISCLOSURE in result
    assert "Not verified against a real SEC filing" in result


def test_filings_analysis_without_an_llm_has_no_disclosure_needed():
    # No LLM means no generated content to mislabel -- the offline prompt
    # echo is already self-evidently not a real review.
    result = filings_analysis("AAPL", llm=None)
    assert "Offline Filings Prompt" in result
    assert FILING_DISCLOSURE not in result


def test_research_appends_unverified_disclosure():
    result = research("AAPL", llm=_FakeLLM())
    assert "Some analysis text." in result
    assert RESEARCH_DISCLOSURE in result
    assert "General knowledge only, not researched" in result


def test_research_without_an_llm_has_no_disclosure_needed():
    result = research("AAPL", llm=None)
    assert "Offline Research Summary" in result
    assert RESEARCH_DISCLOSURE not in result
