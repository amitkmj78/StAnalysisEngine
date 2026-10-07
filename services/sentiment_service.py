from concurrent.futures import ThreadPoolExecutor

from services.llm_setup import invoke_with_fallback
from services.web_search import search_summary, search_text

SENTIMENT_LABELS = ("Bullish", "Neutral", "Bearish")

# Each ticker does 2 web searches + an LLM call — bounded the same way as
# the other multi-ticker fan-outs in this codebase (e.g. index_fund_service's
# MAX_PARALLEL_FETCHES) rather than importing that constant across an
# unrelated domain boundary.
MAX_PARALLEL_SENTIMENT_FETCHES = 4


def get_sentiment_summary(ticker: str, llms: list | None = None) -> tuple[str, list[dict]]:
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

    Returns (context_text, sources) -- NFR-5: sources is the real,
    structured source list search_summary now returns (title + url per
    result), not re-parsed out of the LLM's own prose. Empty without
    `llms`, since search_text already keeps each result's URL inline in
    the text itself rather than needing a separate structured list.
    """
    sections = []
    sources: list[dict] = []

    def _fetch(query: str) -> str:
        if llms:
            text, found = search_summary(query, llms)
            sources.extend(found)
            return text
        return search_text(query)

    try:
        news_results = _fetch(f"{ticker} stock recent news headlines and outlook")
        sections.append(f"Recent News for {ticker}:\n\n{news_results}")
    except Exception as e:
        sections.append(f"Error fetching recent news for {ticker}: {e}")

    try:
        earnings_results = _fetch(
            f"{ticker} latest quarterly earnings report EPS revenue guidance analyst reaction"
        )
        sections.append(f"Recent Earnings Context for {ticker}:\n\n{earnings_results}")
    except Exception as e:
        sections.append(f"Error fetching earnings context for {ticker}: {e}")

    return "\n\n".join(sections), sources


def score_ticker_sentiment(ticker: str, llms: list) -> dict:
    """
    Today's real news/earnings sentiment for `ticker`, as a single
    Bullish/Neutral/Bearish label with one-sentence reasoning — a
    "current reading" display, deliberately NOT a 5-day/10-day forecast.

    That distinction isn't cosmetic: docs/market-direction-sentiment-
    requirements.md documents a related predictive sentiment signal
    (used as a forward-return input) that failed its own backtest
    validation gate four separate times (9a-9c) — "news sentiment at
    daily horizons is noisy and mean-reverting; treating it as an alpha
    source directly is the most common way these systems fail." This
    function only summarizes what the news/earnings coverage says right
    now; it makes no claim about where the price goes next.

    Returns {"label": one of SENTIMENT_LABELS or None, "reasoning": str
    or None, "sources": list[dict]}. label/reasoning are None (not
    raised, sources empty) on any search or LLM failure, so one ticker's
    bad day doesn't break the rest of a portfolio's sentiment column.
    """
    try:
        context, sources = get_sentiment_summary(ticker, llms=llms)
        prompt = (
            f"Based only on the following real news and earnings context for {ticker}, "
            "classify today's market sentiment toward this stock.\n\n"
            f"{context}\n\n"
            "Respond in exactly this two-line format, nothing else:\n"
            "SENTIMENT: <Bullish, Neutral, or Bearish>\n"
            "REASON: <one concise sentence citing what in the context above drove this>\n\n"
            "When your REASON mentions guidance, be precise about which quarter it's for: "
            "guidance issued on an earnings call is a forecast for the NEXT quarter, not the "
            "quarter whose results were just reported -- never label forward guidance with the "
            "same quarter as the earnings beat/miss it was announced alongside."
        )
        response, _ = invoke_with_fallback(llms, prompt)
    except Exception:
        return {"label": None, "reasoning": None, "sources": []}

    label = None
    reasoning = None
    for line in response.splitlines():
        line = line.strip()
        if line.upper().startswith("SENTIMENT:"):
            candidate = line.split(":", 1)[1].strip().strip(".")
            for known in SENTIMENT_LABELS:
                if known.lower() == candidate.lower():
                    label = known
                    break
        elif line.upper().startswith("REASON:"):
            reasoning = line.split(":", 1)[1].strip()

    if label is None:
        return {"label": None, "reasoning": None, "sources": []}
    return {"label": label, "reasoning": reasoning, "sources": sources}


def score_tickers_sentiment(tickers: list[str], llms: list) -> dict[str, dict]:
    """Bulk version of score_ticker_sentiment, fanned out across a bounded
    thread pool (each ticker's searches/LLM call is independent I/O) —
    same shape as fund_comparison_service.rank_funds_by_inception's use of
    ThreadPoolExecutor for a per-ticker fetch. Returns {ticker: {"label":
    ..., "reasoning": ...}}, one entry per input ticker (never dropped),
    with score_ticker_sentiment's own None/None fallback preserved for any
    ticker whose search or LLM call failed."""
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SENTIMENT_FETCHES) as executor:
        results = list(executor.map(lambda t: score_ticker_sentiment(t, llms), tickers))
    return dict(zip(tickers, results))
