import datetime

from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.financialAgent import financial_analysis
from Agent.filingAgent import filings_analysis
from Agent.newAgent import news_summary
from Agent.reasearchAgent import research

today_date = datetime.date.today()


def recommend(company_stock: str, llm=None) -> str:
    """
    Synthesize basic info, technical, financial, filings, news, and
    research analyses into one Investment Recommendation Report.

    Requires an LLM to actually synthesize the report; without one, this
    returns the raw gathered context instead of guessing at a report.
    """
    try:
        basic_info = get_basic_stock_info(company_stock)
    except Exception as e:
        basic_info = f"Error in basic info: {e}"

    try:
        technical_info = get_technical_analysis(company_stock)
    except Exception as e:
        technical_info = f"Error in technical analysis: {e}"

    try:
        financial_info = financial_analysis(company_stock, llm=llm)
    except Exception as e:
        financial_info = f"Error in financial analysis: {e}"

    try:
        filings_info = filings_analysis(company_stock, llm=llm)
    except Exception as e:
        filings_info = f"Error in filings analysis: {e}"

    try:
        news_info = news_summary(company_stock, llm=llm)
    except Exception as e:
        news_info = f"Error in news analysis: {e}"

    try:
        research_info = research(company_stock=company_stock, llm=llm)
    except Exception as e:
        research_info = f"Error in research analysis: {e}"

    context = f"""
[BASIC INFO]
{basic_info}

[TECHNICAL ANALYSIS]
{technical_info}

[FINANCIAL ANALYSIS]
{financial_info}

[FILINGS ANALYSIS]
{filings_info}

[NEWS ANALYSIS]
{news_info}

[RESEARCH ANALYSIS]
{research_info}
"""

    prompt = f"""
You are an AI investment analyst. Using ONLY the research context below for
{company_stock} as of {today_date}, write a structured Investment
Recommendation Report with these sections:

1. Executive Summary (3-5 sentences)
2. Financial Analysis (key metrics, balance sheet health, cash flow trends)
3. Market Sentiment (recent price action, news sentiment)
4. Qualitative Insights (filings, risks, strategic changes)
5. Final Recommendation — an explicit Buy / Hold / Sell stance, the
   rationale behind it, a suggested time horizon, and key risks to monitor.

Write one cohesive report; do not mention that this came from separate
tools. Do not invent numbers that are not present in the context below.

Context:
{context}
"""

    if llm is None:
        return (
            f"[Offline Recommendation Context for {company_stock}]\n\n{context}\n\n"
            "Note: No LLM was provided, so this shows the gathered research "
            "instead of a synthesized recommendation."
        )

    try:
        result = llm.invoke(prompt)
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"[RecommendAgent LLM error: {e}]"
