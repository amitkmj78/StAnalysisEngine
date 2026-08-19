from services.web_search import search_summary, search_text


def get_sentiment_summary(ticker: str, llms: list | None = None) -> str:
    """
    Self-hosted search (services.web_search — DuckDuckGo + real content
    extraction) for recent news and earnings context for the ticker, as
    two separate searches — a single generic "sentiment" query tends to
    miss earnings-specific results (EPS, guidance, analyst reaction),
    which is usually exactly what's driving a large single-day or
    after-hours move.

    When `llms` (an ordered list of available providers) is given, each
    search's raw multi-source results are run through an LLM
    summarization pass (services.web_search.search_summary, which tries
    each llm in turn — see services.llm_setup.invoke_with_fallback)
    before being returned — deduplicates overlapping headlines and
    strips ad/promotional noise that otherwise ends up verbatim in
    whatever narrative prompt this text feeds into next. Without llms,
    falls back to the raw formatted text (search_text), same behavior
    as before summarization existed — this keeps the function usable by
    any caller that doesn't have an LLM in scope.
    """
    sections = []
    fetch = (lambda q: search_summary(q, llms)) if llms else search_text

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
