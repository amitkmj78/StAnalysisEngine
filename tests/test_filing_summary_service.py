from services.filing_summary_service import summarize_filing


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


_CURRENT = {
    "form": "10-K",
    "filing_date": "2025-10-31",
    "report_date": "2025-09-27",
    "sections": {
        "Item 1A": "Global supply chain disruptions could increase costs.",
        "Item 7": "Revenue increased 8% year over year, driven by strong iPhone demand.",
    },
}

_PREVIOUS = {
    "form": "10-K",
    "filing_date": "2024-11-01",
    "report_date": "2024-09-28",
    "sections": {
        "Item 1A": "Macroeconomic conditions could materially harm the business.",
        "Item 7": "Revenue increased 2% year over year on steady Mac and iPad demand.",
    },
}


def test_summarize_filing_with_previous_includes_both_dates_and_guardrail():
    llm = _FakeLLM("Revenue growth accelerated from 2% to 8%, driven by iPhone demand.")
    result = summarize_filing([llm], "AAPL", _CURRENT, _PREVIOUS)

    assert result["summary"] == "Revenue growth accelerated from 2% to 8%, driven by iPhone demand."
    prompt = llm.last_prompt
    assert "2025-10-31" in prompt and "2024-11-01" in prompt
    assert "do not invent any numbers" in prompt.lower()
    assert "iPhone demand" in prompt and "Mac and iPad demand" in prompt
    assert "2024-11-01" in result["method"]
    assert "Item 1A" in result["method"] and "Item 7" in result["method"]


def test_summarize_filing_without_previous_discloses_no_prior_filing():
    llm = _FakeLLM("This is Apple's first tracked 10-K; revenue grew driven by iPhone demand.")
    result = summarize_filing([llm], "AAPL", _CURRENT, None)

    prompt = llm.last_prompt
    assert "No prior 10-K filing is on file" in prompt
    assert "2024-11-01" not in prompt  # nothing from _PREVIOUS should leak in
    assert "No prior 10-K filing is on file" in result["method"]


def test_summarize_filing_returns_none_when_all_providers_fail():
    assert summarize_filing([_RaisingLLM()], "AAPL", _CURRENT, _PREVIOUS) is None


def test_summarize_filing_method_notes_truncation():
    truncated_current = {
        **_CURRENT,
        "sections": {"Item 7": "x" * 15000},  # == MAX_SECTION_CHARS, i.e. truncated
    }
    llm = _FakeLLM("Summary text.")
    result = summarize_filing([llm], "AAPL", truncated_current, None)
    assert "Truncated" in result["method"]
    assert "Item 7" in result["method"]
