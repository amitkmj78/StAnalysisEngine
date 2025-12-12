# Agent/reasearchAgent.py

"""
Research Agent
--------------
This module provides a lightweight wrapper used by your meta-agent
to perform deep qualitative research on a stock.

IMPORTANT:
- This version does NOT depend on get_research_llm().
- LLM must be passed in from app.py or handled at agent level.
"""

def research(company_stock: str, user_prompt: str | None = None, llm=None):
    """
    Perform qualitative long-form research about a stock.

    Parameters:
    - company_stock (str): The stock ticker.
    - user_prompt (str, optional): Custom override prompt.
    - llm: Optional LLM instance to produce real text.

    If llm is provided → returns real LLM output.
    If llm is None → returns a placeholder summary.
    """

    base_prompt = f"""
Provide deep qualitative research insights for stock {company_stock}.

Focus on:
- Competitive landscape
- Market position
- Key risks and vulnerabilities
- Strategic advantages
- Long-term growth opportunities
- Macro factors impacting the business
- Industry outlook

Write in 2–4 short paragraphs, clear and professional.
"""

    final_prompt = user_prompt if user_prompt else base_prompt

    # If the LLM is available, use it
    if llm:
        try:
            result = llm.invoke(final_prompt)
            return getattr(result, "content", str(result))
        except Exception as e:
            return f"[ResearchAgent LLM error: {e}]"

    # Fallback if no LLM provided
    return (
        f"[Offline Research Summary for {company_stock}]\n\n"
        f"{final_prompt}\n\n"
        "Note: No LLM was provided to generate detailed analysis."
    )
