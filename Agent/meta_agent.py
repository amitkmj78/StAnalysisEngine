import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage


# ---------------- TOOL DEFINITIONS ---------------- #

@tool
def company_basics(ticker: str):
    """Return basic company profile"""
    from Agent.basicAgent import get_basic_stock_info
    return get_basic_stock_info(ticker)


@tool
def technical_analysis(ticker: str):
    """Return RSI & MACD analysis"""
    from Agent.technicalAgent import get_technical_analysis
    return get_technical_analysis(ticker)


@tool
def financial_analysis_tool(ticker: str):
    """Valuation metrics & profitability"""
    from Agent.financialAgent import financial_analysis
    return financial_analysis(ticker)


@tool
def filings_analysis_tool(ticker: str):
    """SEC filings risk summary"""
    from Agent.filingAgent import filings_analysis
    return filings_analysis(ticker)


@tool
def news_sentiment(ticker: str):
    """Market sentiment from news"""
    from Agent.newAgent import news_summary
    return news_summary(ticker)


@tool
def research_report(ticker: str):
    """Full equity research report"""
    from Agent.reasearchAgent import research
    return research(company_stock=ticker)


@tool
def final_recommendation(ticker: str):
    """Buy / Hold / Sell decision"""
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


# ---------------- META AGENT BUILDER ---------------- #

def build_agent(llm: ChatOpenAI):
    system_prompt = """
    You are a Wall Street equity analyst.
    - ALWAYS call at least 3 tools before making investment advice
    - Use evidence to justify your conclusions
    - Be accurate, confident & concise
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Bind tools to LLM — it decides which to call
    agent = prompt | llm.bind_tools(TOOLS)

    return agent


# ---------------- EXECUTION LOGIC ---------------- #

def ask_meta_agent(agent, ticker: str, user_question: str) -> str:

    full_query = (
        f"Ticker: {ticker}\n"
        f"Use tools before answering.\n"
        f"Question: {user_question}"
    )

    try:
        response = agent.invoke({"input": full_query})

        # Tool-executed output comes in AIMessage
        if isinstance(response, AIMessage) and response.content:
            return response.content

        return str(response)

    except Exception as e:
        st.error(f"❌ Meta-Agent Error: {e}")
        return "⚠️ AI could not complete this financial analysis."
