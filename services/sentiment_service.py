from langchain_community.tools.tavily_search import TavilySearchResults

from .config import TAVILY_API_KEY


def get_sentiment_summary(ticker: str) -> str:
    """
    Use Tavily to fetch a quick sentiment/news context for the ticker.
    """
    if not TAVILY_API_KEY:
        return "Sentiment analysis unavailable: TAVILY_API_KEY not configured."

    tavily = TavilySearchResults(api_key=TAVILY_API_KEY)
    try:
        search_results = tavily.run(
            f"{ticker} stock market news sentiment analysis, recent headlines, and outlook"
        )
        return f"News & Sentiment Context for {ticker}:\n\n{search_results}"
    except Exception as e:
        return f"Error during Tavily sentiment search: {e}"
