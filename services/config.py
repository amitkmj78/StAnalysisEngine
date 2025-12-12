# services/config.py
import datetime
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# WebSocket-related
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
WS_PRICE_FEED_URL = os.getenv("WS_PRICE_FEED_URL")  # optional custom feed URL


def validate_environment():
    """Validate env config and show helpful messages in the UI."""
    if not GROQ_API_KEY and not OPENAI_API_KEY:
        st.error(
            "❌ No LLM API keys configured.\n\n"
            "Please set at least one of these in your .env or environment:\n"
            "- GROQ_API_KEY for Llama3 via Groq\n"
            "- OPENAI_API_KEY for OpenAI GPT models"
        )
        st.stop()

    if not TAVILY_API_KEY:
        st.warning(
            "⚠️ TAVILY_API_KEY is not set. Web search tools (Tavily) will be limited.\n"
            "Set TAVILY_API_KEY in your .env to enable full research & sentiment features."
        )

    if not FINNHUB_API_KEY and not WS_PRICE_FEED_URL:
        st.warning(
            "⚠️ Real-time WebSocket feed is not configured.\n"
            "Set either FINNHUB_API_KEY or WS_PRICE_FEED_URL in your .env "
            "to enable the Live Price tab."
        )
    st.info(f"Today's Date: {datetime.date.today()}")
