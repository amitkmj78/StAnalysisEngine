import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from services.web_search import search_text


def _get_llm():
    """Return whichever LLM is available (Groq or OpenAI) using valid models."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

    return None



def news_summary(ticker: str, llm=None) -> str:
    """
    Fetch real news using this app's own self-hosted search
    (services.web_search — DuckDuckGo + real content extraction, replaces
    the Tavily-backed version this used to be), then summarize & analyze
    with LLM. Returns a detailed professional news intelligence brief.
    """

    query = (
        f"Latest breaking news, earnings report results (EPS, revenue, guidance), and other "
        f"market-moving headlines about {ticker} stock. Summarize factual content only."
    )

    try:
        raw_news = search_text(query)
        if not raw_news:
            return f"No recent news found for {ticker}."
    except Exception as e:
        return f"Error fetching news: {e}"

    # Step 2 — Use LLM to analyze & summarize
    llm_to_use = llm or _get_llm()
    if llm_to_use is None:
        return raw_news  # fallback to raw text

    prompt = f"""
You are a financial news analyst.

Analyze the following raw news results about **{ticker}**:

========================
{raw_news}
========================

Produce a **professional, structured intelligence summary** including:

1. **Top Headlines (bullet summary)**
2. **Earnings & Financial Results** — most recent quarterly earnings if mentioned: EPS/revenue vs.
   estimates, guidance changes, and analyst reaction. Say explicitly if no earnings news is present.
3. **Market Impact** — does anything here plausibly explain a large recent price move?
4. **Key Risks Discussed in News**
5. **Positive Catalysts / Opportunities**
6. **Sentiment Score (-1 to +1)**
7. **Overall Outlook in 3–5 sentences**

Be concise but insightful.
Do NOT hallucinate — use only info from the news.
"""

    try:
        result = llm_to_use.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        return f"[LLM Error] {e}"
