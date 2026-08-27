def format_analysis_record(
    ticker: str,
    analysis_type: str,
    content: str,
    confidence: str = "medium"
) -> str:
    return f"""
Ticker: {ticker}
Analysis Type: {analysis_type}
Confidence: {confidence}

Analysis:
{content}
""".strip()
