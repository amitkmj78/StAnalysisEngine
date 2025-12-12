import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


def _get_llm():
    """Return whichever LLM is available (Groq or OpenAI) using valid models."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

    return None



def news_summary(ticker: str) -> str:
    """
    Fetch real news using Tavily API, then summarize & analyze with LLM.
    Returns a detailed professional news intelligence brief.
    """

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return f"Tavily API key missing. Cannot fetch news for {ticker}."

    try:
        # Step 1 — Fetch real news from Tavily
        tavily = TavilySearchResults(api_key=tavily_key)

        query = f"Latest breaking news, financial updates, and market-moving headlines about {ticker} stock. Summarize factual content only."

        raw_news = tavily.run(query)

        if not raw_news:
            return f"No recent news found for {ticker}."

    except Exception as e:
        return f"Error fetching news: {e}"

    # Step 2 — Use LLM to analyze & summarize
    llm = _get_llm()
    if llm is None:
        return raw_news  # fallback to raw text

    prompt = f"""
You are a financial news analyst.

Analyze the following raw news results about **{ticker}**:

========================
{raw_news}
========================

Produce a **professional, structured intelligence summary** including:

1. **Top Headlines (bullet summary)**
2. **Market Impact**
3. **Key Risks Discussed in News**
4. **Positive Catalysts / Opportunities**
5. **Sentiment Score (-1 to +1)**
6. **Overall Outlook in 3–5 sentences**

Be concise but insightful.
Do NOT hallucinate — use only info from the news.
"""

    try:
        result = llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        return f"[LLM Error] {e}"
