import datetime

today_date = datetime.date.today()


def filings_analysis(company_stock: str, llm=None) -> str:
    """
    Ask the LLM to reason about recent 10-Q/10-K filings for a stock.

    This does not fetch real EDGAR filings — it relies on the LLM's own
    training knowledge, so it can be stale or incomplete for recent
    filings. Requires an LLM; without one, returns the analysis prompt
    itself rather than guessing at filing contents.
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
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"[FilingsAgent LLM error: {e}]"
