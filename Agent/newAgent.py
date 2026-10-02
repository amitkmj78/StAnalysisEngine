import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from services.web_search import format_results, search


def _get_llm():
    """Return whichever LLM is available (Groq or OpenAI) using valid models."""
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4
        )

    return None


def _fetch_and_summarize(ticker: str, llm=None) -> tuple[str, list[dict]]:
    """Shared by news_summary and news_summary_with_sources: fetches real
    news via this app's own self-hosted search (services.web_search —
    DuckDuckGo + real content extraction) and summarizes it with an LLM.
    Returns (summary_text, sources) where sources is the real
    [{"title", "url"}, ...] list taken directly from the structured search
    response -- ASK-1's citations rely on this list, not on the LLM
    preserving URLs through its own summarization (it reliably doesn't)."""
    query = (
        f"Latest breaking news, earnings report results (EPS, revenue, guidance), and other "
        f"market-moving headlines about {ticker} stock. Summarize factual content only."
    )

    try:
        response = search(query, max_results=5, include_raw_content=False)
        if not response.results:
            return f"No recent news found for {ticker}.", []
        raw_news = format_results(response)
    except Exception as e:
        return f"Error fetching news: {e}", []

    sources = [{"title": r.title, "url": r.url} for r in response.results]

    # Step 2 — Use LLM to analyze & summarize
    llm_to_use = llm or _get_llm()
    if llm_to_use is None:
        return raw_news, sources  # fallback to raw text (still carries URLs inline)

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
        summary = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        summary = f"[LLM Error] {e}"

    return summary, sources


def news_summary(ticker: str, llm=None) -> str:
    """
    Fetch real news and summarize & analyze with LLM. Returns a detailed
    professional news intelligence brief. Unchanged contract for existing
    callers (Agent/recommendAgent.py, services/analysis_service.py) —
    see news_summary_with_sources for the version that also returns real
    article URLs.
    """
    summary, _sources = _fetch_and_summarize(ticker, llm=llm)
    return summary


def news_summary_with_sources(ticker: str, llm=None) -> dict:
    """Same summary as news_summary, plus the real article sources
    ({"title", "url"}) taken straight from the search response -- used by
    Agent/meta_agent.py's news_sentiment tool so citations in the final
    chat answer are guaranteed-accurate URLs, not LLM-reconstructed ones."""
    summary, sources = _fetch_and_summarize(ticker, llm=llm)
    return {"summary": summary, "sources": sources}
