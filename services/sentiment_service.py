from langchain_community.tools.tavily_search import TavilySearchResults

from .config import TAVILY_API_KEY


def get_sentiment_summary(ticker: str) -> str:
    """
    Use Tavily to fetch recent news and earnings context for the ticker, as
    two separate searches — a single generic "sentiment" query tends to miss
    earnings-specific results (EPS, guidance, analyst reaction), which is
    usually exactly what's driving a large single-day or after-hours move.
    """
    if not TAVILY_API_KEY:
        return "News/earnings context unavailable: TAVILY_API_KEY not configured."

    tavily = TavilySearchResults(api_key=TAVILY_API_KEY)
    sections = []

    try:
        news_results = tavily.run(f"{ticker} stock recent news headlines and outlook")
        sections.append(f"Recent News for {ticker}:\n\n{news_results}")
    except Exception as e:
        sections.append(f"Error fetching recent news for {ticker}: {e}")

    try:
        earnings_results = tavily.run(
            f"{ticker} latest quarterly earnings report EPS revenue guidance analyst reaction"
        )
        sections.append(f"Recent Earnings Context for {ticker}:\n\n{earnings_results}")
    except Exception as e:
        sections.append(f"Error fetching earnings context for {ticker}: {e}")

    return "\n\n".join(sections)
