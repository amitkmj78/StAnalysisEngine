from services.earnings_release_service import NO_QA_CAVEAT, summarize_earnings_release


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return _FakeMessage(self.response_text)


class _RaisingLLM:
    def invoke(self, prompt: str):
        raise RuntimeError("provider unavailable")


_RELEASE_TEXT = (
    "Apple today announced financial results for its fiscal 2026 third quarter. "
    "The Company posted revenue of $90.0 billion, up 8 percent year over year, and quarterly EPS of $1.65. "
    "\"We are confident in our product pipeline heading into the holiday quarter,\" said the CEO."
)


def test_summarize_earnings_release_includes_date_and_guardrail():
    llm = _FakeLLM("Revenue was $90.0B, up 8% YoY; EPS $1.65. Management expressed confidence in the pipeline.")
    result = summarize_earnings_release([llm], "AAPL", "2026-07-30", _RELEASE_TEXT)

    assert "$90.0B" in result["summary"]
    prompt = llm.last_prompt
    assert "2026-07-30" in prompt
    assert "do not invent any numbers" in prompt.lower()
    assert "$90.0 billion" in prompt


def test_summarize_earnings_release_prompt_forbids_fabricating_qa():
    """The single most important guardrail for this feature: the source
    text never contains analyst Q&A, so the prompt must explicitly forbid
    inventing any, not just rely on the generic don't-invent instruction."""
    llm = _FakeLLM("Summary text.")
    summarize_earnings_release([llm], "AAPL", "2026-07-30", _RELEASE_TEXT)
    prompt = llm.last_prompt
    assert "does not include analyst" in prompt.lower() or "does not include analyst q&a" in prompt.lower()
    assert "never describe analyst questions" in prompt.lower()


def test_summarize_earnings_release_method_discloses_edgar_only_scope():
    llm = _FakeLLM("Summary text.")
    result = summarize_earnings_release([llm], "AAPL", "2026-07-30", _RELEASE_TEXT)
    assert NO_QA_CAVEAT in result["method"]
    assert "not a transcript" in result["method"]
    assert "2026-07-30" in result["method"]


def test_summarize_earnings_release_returns_none_when_all_providers_fail():
    assert summarize_earnings_release([_RaisingLLM()], "AAPL", "2026-07-30", _RELEASE_TEXT) is None


def test_summarize_earnings_release_notes_truncation():
    long_text = "x" * 4000  # == MAX_SECTION_CHARS
    llm = _FakeLLM("Summary text.")
    result = summarize_earnings_release([llm], "AAPL", "2026-07-30", long_text)
    assert "[TRUNCATED]" in result["method"]
