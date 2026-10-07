import datetime

today_date = datetime.date.today()

# NFR-5: this tool does not fetch real EDGAR filings -- the model answers
# from its own training knowledge, which can be stale, wrong, or just
# invented for a company/quarter it was never actually trained on. The
# real, grounded EDGAR fetch (services/filing_summary_service.py, used
# for SUM-1's filing summaries) isn't wired into this sync, tool-calling
# path -- a bigger integration (this agent is called synchronously;
# filing_summary_service's EDGAR fetch is async) than a disclosure fix
# warrants on its own. Until that's done, every response here is labeled
# as unverified rather than presented as a real filings review, so "no
# invented numbers" isn't silently violated.
UNVERIFIED_DISCLOSURE = (
    "\n\n⚠ Not verified against a real SEC filing — this is the model's own recollection, "
    "which can be stale, incomplete, or wrong. For an actually-fetched filing, see the "
    "Filing Summaries panel on the stock's own page."
)


def filings_analysis(company_stock: str, llm=None) -> str:
    """
    Ask the LLM to reason about recent 10-Q/10-K filings for a stock.

    This does not fetch real EDGAR filings — it relies on the LLM's own
    training knowledge, so it can be stale, incomplete, or invented for
    recent filings. Every real response carries UNVERIFIED_DISCLOSURE so
    this is never mistaken for a grounded review. Requires an LLM;
    without one, returns the analysis prompt itself rather than guessing
    at filing contents.
    """
    prompt = f"""
Analyze the latest 10-Q and 10-K filings from EDGAR for the stock
{company_stock} as of today {today_date}. Focus on key sections like
Management's Discussion and Analysis, financial statements, insider
trading activity, and any disclosed risks. Extract relevant data and
insights that could influence the stock's future performance.

Produce an expanded report that highlights significant findings from
these filings, including any red flags or positive indicators for the
investor. If you are not confident about the specifics of the most
recent filing, say so explicitly rather than guessing.
"""

    if llm is None:
        return (
            f"[Offline Filings Prompt for {company_stock}]\n\n{prompt}\n\n"
            "Note: No LLM was provided, so this shows the analysis prompt "
            "instead of a real filings review."
        )

    try:
        result = llm.invoke(prompt)
        content = getattr(result, "content", str(result))
        return content + UNVERIFIED_DISCLOSURE
    except Exception as e:
        return f"[FilingsAgent LLM error: {e}]"
