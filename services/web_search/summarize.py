"""
Optional LLM summarization pass over raw search results — a "better
picture" version of client.py's search_text(), at the cost of one extra
LLM call per query. Raw multi-source search content is noisy in
practice: near-duplicate headlines across sources, ad/promotional
boilerplate mixed into extracted article text, truncated snippets. This
condenses that into a clean, deduplicated, fact-focused brief before a
downstream narrative/analysis LLM call ever sees it.

Deliberately NOT the default behavior of search_text() — this is an
opt-in for callers that already have an LLM in scope and want the
tradeoff (see services/sentiment_service.py, the first caller to use
it). Takes `llms` (an ordered list, tried in turn via
services.llm_setup.invoke_with_fallback) as a parameter rather than
resolving providers itself, matching this codebase's existing
convention (e.g. services/portfolio_alert_service.py's
build_drop_analysis(llms, ...)) of injecting the LLMs at the call site
rather than each service layer re-implementing its own LLM selection.
"""

from typing import Optional

from services.llm_setup import invoke_with_fallback

from .client import SearchResponse, format_results, search


def summarize_results(response: SearchResponse, llms: list, focus: Optional[str] = None) -> str:
    """
    Falls back to the raw formatted text (never raises) if every LLM in
    `llms` fails, or there are no results to summarize — a
    summarization hiccup must not block whatever narrative/analysis is
    waiting on this.
    """
    raw_text = format_results(response)
    if not response.results:
        return raw_text

    focus_line = f" Focus specifically on: {focus}." if focus else ""
    prompt = (
        f'Below are raw web search results for the query "{response.query}".{focus_line}\n\n'
        f"{raw_text}\n\n"
        "Summarize this into a clean, factual brief for a stock analyst. Rules:\n"
        "- Deduplicate overlapping headlines/facts across sources — state each fact once.\n"
        "- Drop advertisements, promotional content, and anything not about the actual company/stock.\n"
        "- Keep concrete facts (numbers, dates, named catalysts) — do not vague-ify them.\n"
        "- If sources disagree or a fact is unconfirmed, say so rather than silently picking one.\n"
        "- 4-8 sentences. No preamble, no closing remarks — just the brief."
    )
    try:
        content, _ = invoke_with_fallback(llms, prompt)
        return content
    except Exception:
        return raw_text


def search_summary(query: str, llms: list, max_results: int = 5) -> str:
    """Search + LLM-summarize in one call."""
    response = search(query, max_results=max_results, include_raw_content=False)
    return summarize_results(response, llms, focus=query)
