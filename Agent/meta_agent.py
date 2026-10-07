# ============================================================
# meta_agent.py — FINAL STABLE VERSION WITH FULL DEBUG LOGGING
# ============================================================

import datetime
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

DEBUG = True     # turn OFF by setting to False
DEBUG_LOGS = []  # store logs for Streamlit UI


# ------------------------------------------------------------
# UTILITY: DEBUG LOGGER
# ------------------------------------------------------------
def log_debug(title: str, data: str):
    if DEBUG:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        DEBUG_LOGS.append(f"\n========== DEBUG @ {timestamp} — {title} ==========\n{data}\n")


def get_debug_logs():
    return "\n".join(DEBUG_LOGS)


# ASK-2: the one explicit "don't guess" guardrail instruction, reused both
# in the tool-calling agent's own system prompt and appended to the
# post-tool summarization prompt in ask_meta_agent (that step is a second,
# independent LLM call with its own instructions, so it needs the same
# guardrail restated, not inherited). Mirrors the existing precedent in
# Agent/filingAgent.py's "say so explicitly rather than guessing" line.
DONT_GUESS_INSTRUCTION = (
    "If the information available doesn't actually contain enough to answer the "
    "specific question asked, say so plainly (e.g. \"I don't know\" or \"I don't have "
    "enough data to answer that\") instead of guessing or relying on general knowledge "
    "that isn't grounded in the data shown."
)

SYSTEM_PROMPT = f"""
    You are a Wall Street equity analyst.
    RULES:
    - You MUST call at least 1 tool when relevant.
    - You MAY call multiple tools when necessary.
    - After tool calls, return a final investor-ready explanation grounded in those tool
      results.
    - NEVER output JSON unless asked.
    - Write concise, readable English.
    - {DONT_GUESS_INSTRUCTION}
    """


# ------------------------------------------------------------
# BUILD META-AGENT
# ------------------------------------------------------------
def build_agent(llm):
    """
    Build the meta-agent's orchestration chain and its tools.

    The tools are defined here (not at module level) so each one closes
    over `llm` — the same model the user picked in the sidebar is what
    actually answers financial/filings/news/research/recommendation
    sub-questions, instead of those tools silently hardcoding their own
    model regardless of the user's selection.

    `state["sources"]` is a mutable container the news_sentiment tool
    below fills in with its real, structured source list (NFR-5) --
    read back by ask_meta_agent_with_sources after a call, since a
    LangChain @tool can only return a string to the model itself.
    """
    state: dict = {"sources": []}

    @tool
    def company_basics(ticker: str):
        """Return basic company profile such as name, sector, industry, and price."""
        from Agent.basicAgent import get_basic_stock_info
        return get_basic_stock_info(ticker)

    @tool
    def technical_analysis(ticker: str):
        """Return technical indicators such as RSI and MACD signals."""
        from Agent.technicalAgent import get_technical_analysis
        return get_technical_analysis(ticker)

    @tool
    def financial_analysis_tool(ticker: str):
        """Return valuation metrics, profitability, ROE, DuPont, and financial strength."""
        from Agent.financialAgent import financial_analysis
        return financial_analysis(ticker, llm=llm)

    @tool
    def filings_analysis_tool(ticker: str):
        """Return SEC filings analysis including 10-K risk factors and red flags."""
        from Agent.filingAgent import filings_analysis
        return filings_analysis(ticker, llm=llm)

    @tool
    def news_sentiment(ticker: str):
        """Return recent news, earnings results/guidance, and headline sentiment score for a
        ticker. Use this whenever the user asks what's driving the stock's price, including
        explaining a large recent move or an after-hours/pre-market jump."""
        from Agent.newAgent import news_summary_with_sources
        result = news_summary_with_sources(ticker, llm=llm)
        state["sources"] = result["sources"]
        # ASK-1: real article URLs, taken straight from the structured search
        # response (never re-parsed out of the LLM's summary text, which
        # drops/mangles them) and appended here in plain, non-LLM-generated
        # text so the final answer can cite them verbatim.
        if result["sources"]:
            sources_block = "\n".join(f"- {s['title']}: {s['url']}" for s in result["sources"])
            return f"{result['summary']}\n\nSources:\n{sources_block}"
        return result["summary"]

    @tool
    def research_report(ticker: str):
        """Return full professional equity research report with catalysts and risks."""
        from Agent.reasearchAgent import research
        return research(ticker, llm=llm)

    @tool
    def final_recommendation(ticker: str):
        """Return final Buy / Hold / Sell recommendation."""
        from Agent.recommendAgent import recommend
        return recommend(ticker, llm=llm)

    tools = [
        company_basics,
        technical_analysis,
        financial_analysis_tool,
        filings_analysis_tool,
        news_sentiment,
        research_report,
        final_recommendation,
    ]

    log_debug("BUILD_AGENT — SYSTEM PROMPT", SYSTEM_PROMPT)
    log_debug("BUILD_AGENT — TOOLS LOADED", str([t.name for t in tools]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}")
    ])

    agent_runnable = prompt | llm.bind_tools(tools)
    # `llm` (no tools bound) is kept alongside the tool-bound runnable so
    # ask_meta_agent's post-tool-call summarization step can use a model
    # that's structurally incapable of responding with another tool call
    # instead of text — see the comment at that call site for why this
    # matters.
    return {"agent": agent_runnable, "tools": tools, "llm": llm, "_state": state}


# ------------------------------------------------------------
# MAIN EXECUTION FUNCTION (SAFE)
# ------------------------------------------------------------
def ask_meta_agent(meta_agent, ticker: str, question: str) -> str:
    agent = meta_agent["agent"]
    tools = meta_agent["tools"]
    llm = meta_agent["llm"]
    combined = ""  # set below if any tools actually ran; referenced in the empty-response fallback

    full_input = f"""
    Analyze stock: {ticker}

    User Question:
    {question}

    If needed, call tools.
    After tools, ALWAYS give a final answer.
    """

    log_debug("ASK_META_AGENT — INPUT", full_input)

    try:
        raw_response = agent.invoke({"input": full_input})
        log_debug("ASK_META_AGENT RAW RESPONSE", str(raw_response))
    except Exception as e:
        return f"❌ Meta-agent crashed: {e}"

    # ---------------------------
    # HANDLE TOOL CALL OUTPUTS
    # ---------------------------
    if isinstance(raw_response, AIMessage):
        # contains possible tool calls OR final text
        if raw_response.tool_calls:
            # Agent decided to invoke a tool — must execute manually!
            tool_outputs = []
            failed_tools = []  # ASK-2: tracked separately from tool_outputs so a
            # failure is never summarized as if it were real data (the prior
            # behavior appended "Error executing {tool}: {e}" into the same
            # list as successful results, leaving the summarization LLM no
            # way to tell a fact from a failure).

            for call in raw_response.tool_calls:
                tool_name = call["name"]
                args = call["args"]

                log_debug("TOOL INVOCATION", f"Tool: {tool_name}\nArgs: {args}")

                try:
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    result = tool_fn.run(args)
                    tool_outputs.append(f"Tool {tool_name} result:\n{result}")
                except Exception as e:
                    log_debug("TOOL_FAILED", f"{tool_name}: {e}")
                    failed_tools.append(tool_name)

            combined = "\n\n".join(tool_outputs)
            log_debug("TOOL OUTPUT AGGREGATED", combined)

            if not tool_outputs:
                # Every tool that was called failed -- there is no real data
                # to summarize, and asking the LLM to "summarize" nothing
                # invites it to guess from general knowledge instead. Return
                # an honest answer directly; never invoke the summarization
                # LLM with zero real content.
                return (
                    "I don't have enough data to answer that — "
                    f"{', '.join(failed_tools)} failed to return data for this request."
                )

            failed_note = (
                f"\n\nNote: {', '.join(failed_tools)} failed and returned no data for this "
                "request -- do not guess at what they might have shown."
                if failed_tools else ""
            )

            # Now ask the LLM to summarize tool results. Deliberately
            # `llm.invoke` (no tools bound), not `agent.invoke` — the
            # actual bug this fixes: re-invoking the tool-bound `agent`
            # here let the model decide to call yet another tool instead
            # of summarizing (observed with Groq's gpt-oss-20b), which
            # left `.content` empty and silently fell through to the
            # "empty message" warning below on every such request. A
            # plain LLM call is structurally incapable of returning a
            # tool call, so this always produces text.
            #
            # This is a second, independent LLM call with its own
            # instructions, so the don't-guess guardrail (SYSTEM_PROMPT)
            # and the citation convention (a tool's own "Sources:" block,
            # see news_sentiment above) have to be restated here -- neither
            # is inherited from the first call.
            summary_prompt = f"""
            Summarize the following tool outputs into a final investor-ready answer.
            {DONT_GUESS_INSTRUCTION}
            When a claim relies on a tool's own "Sources:" list, cite at least one of
            those sources (title or URL) in your answer. For any other claim, name
            which tool/data it came from.

            {combined}{failed_note}
            """

            final_msg = llm.invoke(summary_prompt)
            text = final_msg.content
        else:
            text = raw_response.content
    else:
        text = str(raw_response)

    # ---------------------------
    # ENSURE NON-EMPTY RESPONSE
    # ---------------------------
    if not text or text.strip() == "":
        log_debug("EMPTY_RESPONSE_FALLBACK", "Agent returned empty output.")
        # Real tool output beats a bare warning, if any was gathered —
        # only fall back to the warning when there's truly nothing to show.
        if combined.strip():
            return combined
        return "⚠️ Meta-agent responded with an empty message."

    log_debug("ASK_META_AGENT — PARSED TEXT", text)
    return text


def ask_meta_agent_with_sources(meta_agent, ticker: str, question: str) -> tuple[str, list[dict]]:
    """NFR-5: same call as ask_meta_agent, but also returns the real,
    structured sources the news_sentiment tool collected this call (if
    it was called) via meta_agent["_state"] -- instead of those sources
    only existing flattened inside the answer's prose. A separate
    function, not a changed return shape on ask_meta_agent itself, so
    its other callers (services/analysis_service.py, services/app.py)
    are untouched.

    Resets _state["sources"] before calling, so a reused meta_agent
    (same build_agent() object across several questions) never leaks a
    PRIOR question's sources into an answer that didn't call the tool
    this time."""
    meta_agent["_state"]["sources"] = []
    text = ask_meta_agent(meta_agent, ticker, question)
    return text, meta_agent["_state"]["sources"]
