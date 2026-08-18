from services.web_search import search_text


def get_sentiment_summary(ticker: str) -> str:
    """
    Self-hosted search (services.web_search — DuckDuckGo + real content
    extraction, replaces the Tavily-backed version this used to be) for
    recent news and earnings context for the ticker, as two separate
    searches — a single generic "sentiment" query tends to miss
    earnings-specific results (EPS, guidance, analyst reaction), which is
    usually exactly what's driving a large single-day or after-hours move.
    """
    sections = []

    try:
        news_results = search_text(f"{ticker} stock recent news headlines and outlook")
        sections.append(f"Recent News for {ticker}:\n\n{news_results}")
    except Exception as e:
        sections.append(f"Error fetching recent news for {ticker}: {e}")

    try:
        earnings_results = search_text(
            f"{ticker} latest quarterly earnings report EPS revenue guidance analyst reaction"
        )
        sections.append(f"Recent Earnings Context for {ticker}:\n\n{earnings_results}")
    except Exception as e:
        sections.append(f"Error fetching earnings context for {ticker}: {e}")

    return "\n\n".join(sections)
