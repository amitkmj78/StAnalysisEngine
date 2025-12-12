# ============================================================
# meta_agent.py — FINAL STABLE VERSION WITH FULL DEBUG LOGGING
# ============================================================

import json
import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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


# ------------------------------------------------------------
# TOOL DEFINITIONS (ALL DOCSTRINGS FIXED)
# ------------------------------------------------------------

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
    return financial_analysis(ticker)


@tool
def filings_analysis_tool(ticker: str):
    """Return SEC filings analysis including 10-K risk factors and red flags."""
    from Agent.filingAgent import filings_analysis
    return filings_analysis(ticker)


@tool
def news_sentiment(ticker: str):
    """Return summarized news sentiment and headline sentiment score."""
    from Agent.newAgent import news_summary
    return news_summary(ticker)


@tool
def research_report(ticker: str):
    """Return full professional equity research report with catalysts and risks."""
    from Agent.reasearchAgent import research
    return research(ticker)


@tool
def final_recommendation(ticker: str):
    """Return final Buy / Hold / Sell recommendation."""
    from Agent.recommendAgent import recommend
    return recommend(ticker)


TOOLS = [
    company_basics,
    technical_analysis,
    financial_analysis_tool,
    filings_analysis_tool,
    news_sentiment,
    research_report,
    final_recommendation,
]


# ------------------------------------------------------------
# BUILD META-AGENT
# ------------------------------------------------------------
def build_agent(llm: ChatOpenAI):
    system_prompt = """
    You are a Wall Street equity analyst.
    RULES:
    - You MUST call at least 1 tool when relevant.
    - You MAY call multiple tools when necessary.
    - After tool calls, ALWAYS return a final investor-ready explanation.
    - NEVER return an empty message.
    - NEVER output JSON unless asked.
    - Write concise, readable English.
    """

    log_debug("BUILD_AGENT — SYSTEM PROMPT", system_prompt)
    log_debug("BUILD_AGENT — TOOLS LOADED", str([t.name for t in TOOLS]))

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    agent = prompt | llm.bind_tools(TOOLS)
    return agent


# ------------------------------------------------------------
# MAIN EXECUTION FUNCTION (SAFE)
# ------------------------------------------------------------
def ask_meta_agent(agent, ticker: str, question: str) -> str:
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

            for call in raw_response.tool_calls:
                tool_name = call["name"]
                args = call["args"]

                log_debug("TOOL INVOCATION", f"Tool: {tool_name}\nArgs: {args}")

                try:
                    tool_fn = next(t for t in TOOLS if t.name == tool_name)
                    result = tool_fn.run(args)
                    tool_outputs.append(f"Tool {tool_name} result:\n{result}")
                except Exception as e:
                    tool_outputs.append(f"Error executing {tool_name}: {e}")

            combined = "\n\n".join(tool_outputs)
            log_debug("TOOL OUTPUT AGGREGATED", combined)

            # Now ask LLM to summarize tool results
            summary_prompt = f"""
            Summarize the following tool outputs into a final investor-ready answer:

            {combined}
            """

            final_msg = agent.invoke({"input": summary_prompt})
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
        return "⚠️ Meta-agent responded with an empty message."

    log_debug("ASK_META_AGENT — PARSED TEXT", text)
    return text
