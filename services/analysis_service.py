from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research

from .sentiment_service import get_sentiment_summary
from .data_service import get_latest_price
import yfinance as yf
import pandas as pd
from typing import Optional, Dict

def get_analyst_targets(ticker: str) -> Optional[Dict[str, float]]:
    """
    Fetch analyst 12-month price target consensus using Yahoo Finance.
    Returns None if unavailable.
    """

    try:
        stock = yf.Ticker(ticker)
        analyst_info = stock.analysis if hasattr(stock, "analysis") else None

        # Fallback: Some tickers provide price target under "info"
        info = stock.info

        if not info:
            return None

        target_high = info.get("targetHighPrice")
        target_mean = info.get("targetMeanPrice")
        target_low = info.get("targetLowPrice")
        current_price = info.get("currentPrice")

        if any(v is None for v in [target_high, target_mean, target_low, current_price]):
            return None

        return {
            "target_high": float(target_high),
            "target_mean": float(target_mean),
            "target_low": float(target_low),
            "current_price": float(current_price),
        }

    except Exception:
        return None
    
def get_analysis(analysis_type: str, ticker: str, research_prompt: str | None = None) -> str:
    """
    Route analysis requests to the appropriate helper function.
    """
    if analysis_type == "Basic Info":
        return get_basic_stock_info(ticker)

    if analysis_type == "Technical Analysis":
        return get_technical_analysis(ticker)

    if analysis_type == "Financial Analysis":
        return financial_analysis(ticker)

    if analysis_type == "Filings Analysis":
        return filings_analysis(ticker)

    if analysis_type == "News Analysis":
        return news_summary(ticker)

    if analysis_type == "Recommend":
        return recommend(ticker)

    if analysis_type == "Sentiment Analysis":
        return get_sentiment_summary(ticker)

    if analysis_type == "Research Analysis":
        # Prompt optional override
        if research_prompt:
            return research(company_stock=ticker, user_prompt=research_prompt)
        return research(company_stock=ticker)

    if analysis_type == "Real-Time Price":
        price = get_latest_price(ticker)
        return f"Real-time price for {ticker}: {price if price is not None else 'N/A'}"

    return f"Unknown analysis type: {analysis_type}"
