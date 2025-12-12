# ===========================
#  ANALYSIS SERVICE (FIXED)
# ===========================
import yfinance as yf
import pandas as pd
from typing import Optional, Dict
import streamlit as st

# ---- Agent Tools ----
from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research  # correct spelling

# ---- Services ----
from .sentiment_service import get_sentiment_summary
from .data_service import get_latest_price

# ---- Meta Agent ----
from Agent.meta_agent import ask_meta_agent


# ===========================
#  ANALYST TARGET PRICE LOOKUP
# ===========================
def get_analyst_targets(ticker: str) -> Optional[Dict[str, float]]:
    """
    Fetch analyst 12-month price targets via Yahoo Finance.
    Returns None if missing.
    """
    try:
        info = yf.Ticker(ticker).info or {}

        high = info.get("targetHighPrice")
        mean = info.get("targetMeanPrice")
        low = info.get("targetLowPrice")
        price = info.get("currentPrice")

        if None in (high, mean, low, price):
            return None

        return {
            "target_high": float(high),
            "target_mean": float(mean),
            "target_low": float(low),
            "current_price": float(price),
        }

    except Exception:
        return None


# ===========================
#  UNIVERSAL ANALYSIS ROUTER
# ===========================
def get_analysis(analysis_type: str, ticker: str, custom_prompt: str = None) -> str:
    """
    Performs ANY analysis by routing through the meta-agent.
    
    analysis_type examples:
      - Basic Info
      - Technical Analysis
      - Financial Analysis
      - Filings Analysis
      - News Analysis
      - Recommend
      - Real-Time Price
      - Sentiment Analysis
      - Research Analysis
    """

    # Ensure agent exists
    agent = st.session_state.get("meta_agent")
    if agent is None:
        return "⚠️ Meta-agent is not initialized. Please restart the app."

    # Build query sent to meta-agent
    if custom_prompt:
        question = f"{analysis_type} for {ticker}. {custom_prompt}"
    else:
        question = f"{analysis_type} for {ticker}"

    # ---- SAFE CALL WRAPPER ----
    try:
        response = ask_meta_agent(agent, ticker, question)

        # If empty or invalid text returned
        if not response or not str(response).strip():
            return "⚠️ Empty response from meta-agent."

        return response

    except Exception as e:
        return f"❌ Error inside get_analysis(): {e}"


# ===========================
#  ANALYST CONSENSUS SUMMARY
# ===========================
def get_analyst_rating_summary(ticker: str) -> Optional[dict]:
    """
    Fetch Buy/Hold/Sell consensus and analyst stats.
    Returns dict or None.
    """
    try:
        info = yf.Ticker(ticker).info

        if not info or "recommendationKey" not in info:
            return None

        recommendation = info.get("recommendationKey", "N/A").title()
        analyst_count = info.get("numberOfAnalystOpinions", 0)

        rec_mean = info.get("recommendationMean")
        if rec_mean:
            # Yahoo scale: 1 strong buy → 5 strong sell
            buy_pct = max(0, min(100, (5 - rec_mean) / 4 * 100))
        else:
            buy_pct = 50.0  # neutral fallback

        return {
            "consensus": recommendation,
            "analyst_count": analyst_count,
            "buy_pct": round(buy_pct, 1),
            "target_mean": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "current_price": info.get("currentPrice"),
        }

    except Exception:
        return None
