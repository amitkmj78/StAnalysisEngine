from langchain_core.messages import AIMessage

from Agent.meta_agent import SYSTEM_PROMPT, ask_meta_agent


class _FakeTool:
    def __init__(self, name, result=None, raises=None):
        self.name = name
        self._result = result
        self._raises = raises

    def run(self, args):
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeAgent:
    """Stands in for `prompt | llm.bind_tools(tools)` — the tool-bound
    runnable. Only used for the *first* invoke in ask_meta_agent; the
    fix under test means it must never be invoked a second time for
    summarization."""

    def __init__(self, first_response):
        self.first_response = first_response
        self.invoke_count = 0

    def invoke(self, _input):
        self.invoke_count += 1
        return self.first_response


class _FakePlainLLM:
    """Stands in for the tool-free `llm` used for summarization."""

    def __init__(self, response_content):
        self.response_content = response_content
        self.invoke_count = 0
        self.last_prompt = None

    def invoke(self, prompt):
        self.invoke_count += 1
        self.last_prompt = prompt
        return AIMessage(content=self.response_content)


def _tool_call_response():
    return AIMessage(
        content="",
        tool_calls=[{"name": "company_basics", "args": {"ticker": "XLK"}, "id": "call-1"}],
    )


def _two_tool_call_response():
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "company_basics", "args": {"ticker": "XLK"}, "id": "call-1"},
            {"name": "news_sentiment", "args": {"ticker": "XLK"}, "id": "call-2"},
        ],
    )


def test_summarization_uses_plain_llm_not_tool_bound_agent():
    """The actual bug: re-invoking the tool-bound agent for the summary
    step let the model call another tool instead of returning text. The
    summarization call must go through the plain llm, not `agent`."""
    tool_call_msg = _tool_call_response()
    agent = _FakeAgent(tool_call_msg)
    llm = _FakePlainLLM("XLK is a large, diversified tech-sector ETF.")
    tools = [_FakeTool("company_basics", "Name: Technology Select Sector SPDR Fund")]

    result = ask_meta_agent({"agent": agent, "tools": tools, "llm": llm}, "XLK", "is this ok to hold")

    assert result == "XLK is a large, diversified tech-sector ETF."
    assert agent.invoke_count == 1  # only the initial call, never re-invoked for summarization
    assert llm.invoke_count == 1


def test_falls_back_to_raw_tool_output_when_summary_is_empty():
    """Regression test for the exact bug report: Groq's gpt-oss-20b
    returning empty content for the summarization step used to produce
    a bare "empty message" warning even though real tool data had
    already been fetched. Real data should win over a useless warning."""
    tool_call_msg = _tool_call_response()
    agent = _FakeAgent(tool_call_msg)
    llm = _FakePlainLLM("")  # simulates the model returning nothing, even without tools bound
    tools = [_FakeTool("company_basics", "Name: Technology Select Sector SPDR Fund, Price: $183.56")]

    result = ask_meta_agent({"agent": agent, "tools": tools, "llm": llm}, "XLK", "is this ok to hold")

    assert "Technology Select Sector SPDR Fund" in result
    assert "empty message" not in result


def test_returns_warning_only_when_truly_nothing_available():
    agent = _FakeAgent(AIMessage(content=""))  # no tool_calls, no content either
    llm = _FakePlainLLM("should not be called")
    result = ask_meta_agent({"agent": agent, "tools": [], "llm": llm}, "XLK", "is this ok to hold")
    assert result == "⚠️ Meta-agent responded with an empty message."
    assert llm.invoke_count == 0  # summarization path never runs when there were no tool calls


def test_direct_text_response_without_tool_calls():
    agent = _FakeAgent(AIMessage(content="XLK looks reasonably diversified."))
    llm = _FakePlainLLM("should not be called")
    result = ask_meta_agent({"agent": agent, "tools": [], "llm": llm}, "XLK", "is this ok to hold")
    assert result == "XLK looks reasonably diversified."
    assert llm.invoke_count == 0


# --- ASK-2: failures must never be laundered into the summarized "data" ---

def test_one_tool_failing_is_noted_separately_not_mixed_into_summarized_data():
    """A failed tool must not become a fabricated-looking fact in `combined`
    -- it's surfaced as a distinct note in the summarization prompt instead,
    alongside the real data from the tool that did succeed."""
    agent = _FakeAgent(_two_tool_call_response())
    llm = _FakePlainLLM("XLK is a diversified tech ETF.")
    tools = [
        _FakeTool("company_basics", result="Name: Technology Select Sector SPDR Fund"),
        _FakeTool("news_sentiment", raises=RuntimeError("search provider down")),
    ]

    result = ask_meta_agent({"agent": agent, "tools": tools, "llm": llm}, "XLK", "is this ok to hold")

    assert result == "XLK is a diversified tech ETF."
    assert llm.invoke_count == 1  # real data existed, so summarization still runs
    prompt = llm.last_prompt
    assert "Technology Select Sector SPDR Fund" in prompt
    assert "Error executing" not in prompt  # old failure-laundering string must be gone
    assert "news_sentiment" in prompt and "do not guess" in prompt.lower()


def test_all_tools_failing_returns_honest_message_without_calling_llm():
    agent = _FakeAgent(_two_tool_call_response())
    llm = _FakePlainLLM("should not be called")
    tools = [
        _FakeTool("company_basics", raises=RuntimeError("yfinance down")),
        _FakeTool("news_sentiment", raises=RuntimeError("search provider down")),
    ]

    result = ask_meta_agent({"agent": agent, "tools": tools, "llm": llm}, "XLK", "is this ok to hold")

    assert "don't have enough data" in result
    assert "company_basics" in result and "news_sentiment" in result
    assert llm.invoke_count == 0  # never summarize when there's nothing real to summarize


def test_system_prompt_instructs_against_guessing():
    assert "don't know" in SYSTEM_PROMPT.lower()
    assert "guess" in SYSTEM_PROMPT.lower()
