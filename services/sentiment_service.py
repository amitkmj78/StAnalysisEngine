from services.web_search import search_summary, search_text


def get_sentiment_summary(ticker: str, llm=None) -> str:
    """
    Self-hosted search (services.web_search — DuckDuckGo + real content
    extraction) for recent news and earnings context for the ticker, as
    two separate searches — a single generic "sentiment" query tends to
    miss earnings-specific results (EPS, guidance, analyst reaction),
    which is usually exactly what's driving a large single-day or
    after-hours move.

    When `llm` is provided, each search's raw multi-source results are
    run through an LLM summarization pass (services.web_search.
    search_summary) before being returned — deduplicates overlapping
    headlines and strips ad/promotional noise that otherwise ends up
    verbatim in whatever narrative prompt this text feeds into next.
    Without an llm, falls back to the raw formatted text (search_text),
    same behavior as before summarization existed — this keeps the
    function usable by any caller that doesn't have an LLM in scope.
    """
    sections = []
    fetch = (lambda q: search_summary(q, llm)) if llm is not None else search_text

    try:
        news_results = fetch(f"{ticker} stock recent news headlines and outlook")
        sections.append(f"Recent News for {ticker}:\n\n{news_results}")
    except Exception as e:
        sections.append(f"Error fetching recent news for {ticker}: {e}")

    try:
        earnings_results = fetch(
            f"{ticker} latest quarterly earnings report EPS revenue guidance analyst reaction"
        )
        sections.append(f"Recent Earnings Context for {ticker}:\n\n{earnings_results}")
    except Exception as e:
        sections.append(f"Error fetching earnings context for {ticker}: {e}")

    return "\n\n".join(sections)
