# ================================================================
#                   AI Stock Analysis Platform
#                   PHASE 1 - Fixed Core System
# ================================================================

import json
import os
import datetime
import time
import threading
import re

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# ======================================================
# WebSocket Setup
# ======================================================

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

FINNHUB_WS = (
    f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    if FINNHUB_API_KEY else None
)

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None

def on_message(ws, message):
    data = json.loads(message)
    if data.get("type") == "trade":
        price = data["data"][0]["p"]
        st.session_state.latest_price = price

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws):
    print("### WebSocket closed ###")

def run_ws(ticker):
    ws = websocket.WebSocketApp(
        FINNHUB_WS,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    def subscribe():
        time.sleep(1)
        ws.send(json.dumps({"type": "subscribe", "symbol": ticker.upper()}))

    threading.Thread(target=subscribe).start()
    ws.run_forever()

def start_price_feed(ticker):
    if "ws_thread" not in st.session_state or not st.session_state.ws_thread.is_alive():
        st.session_state.ws_thread = threading.Thread(target=run_ws, args=(ticker,))
        st.session_state.ws_thread.daemon = True
        st.session_state.ws_thread.start()


# ======================================================
# Agent-like Helper Tools
# ======================================================

from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research


# ======================================================
# Environment Validation
# ======================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY and not OPENAI_API_KEY:
    st.error("❌ Missing LLM Keys: Set GROQ_API_KEY and/or OPENAI_API_KEY in .env")
    st.stop()

if not TAVILY_API_KEY:
    st.warning("⚠ Tavily news search limited — no sentiment API key configured")

st.info(f"Today's Date: {datetime.date.today()}")


# ======================================================
# LLM Setup ❌ FIXED
# ======================================================

llm_openai = None
llm_groq = None

if OPENAI_API_KEY:
    llm_openai = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

if GROQ_API_KEY:
    llm_groq = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY
    )

available_llms = []
if llm_groq:
    available_llms.append("Groq Llama3-70B")
if llm_openai:
    available_llms.append("OpenAI GPT")

select_llm = st.sidebar.selectbox("Select Model", available_llms)
llm = llm_groq if select_llm == "Groq Llama3-70B" else llm_openai


# ======================================================
# Sidebar Inputs
# ======================================================

ticker = st.sidebar.text_input("Enter Ticker:", "AAPL")

if st.sidebar.button("Start Live Price Feed"):
    if FINNHUB_WS:
        st.success("Started Streaming Live Price")
        start_price_feed(ticker)
    else:
        st.warning("Finnhub WebSocket not configured")

analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    [
        "Research Analysis",
        "Basic Info",
        "Technical Analysis",
        "Financial Analysis",
        "Filings Analysis",
        "News Analysis",
        "Recommend",
        "Sentiment Analysis",
        "Real-Time Price",
    ],
)

timeframe_mapping = {
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
}

timeframe = st.sidebar.radio(
    "Timeframe:",
    list(timeframe_mapping.keys()),
    horizontal=True,
)

prediction_days = st.sidebar.slider("Days Ahead", 1, 60, 5)


# ======================================================
# Caching Stock Data Fetch
# ======================================================

@st.cache_data(ttl=300)
def get_stock_data(ticker, period):
    try:
        return yf.Ticker(ticker).history(period=period).dropna()
    except:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_latest_price(ticker):
    df = get_stock_data(ticker, "1d")
    if df.empty:
        return None
    return round(float(df["Close"].iloc[-1]), 2)


# ======================================================
# ML Price Prediction
# ======================================================

def predict_next_price(ticker, period):
    df = get_stock_data(ticker, period)
    if len(df) < 20:
        return None

    df["Index"] = np.arange(len(df))
    df["Lag1"] = df["Close"].shift(1)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df = df.dropna()

    X = df[["Index", "Lag1", "MA5", "MA10"]]
    y = df["Close"]

    model = GradientBoostingRegressor(n_estimators=400, random_state=42)
    model.fit(X, y)

    last = df.iloc[-1]
    new_x = np.array([[last["Index"] + 1, last["Close"], last["MA5"], last["MA10"]]])

    return float(model.predict(new_x)[0])


# ======================================================
# Sentiment + Research
# ======================================================

def get_sentiment(ticker):
    if not TAVILY_API_KEY:
        return "Sentiment unavailable."
    tav = TavilySearchResults(api_key=TAVILY_API_KEY)
    return tav.run(f"{ticker} stock outlook sentiment news")


# ======================================================
# Analysis Router (For Now – Until Multi-Agent in Phase 3)
# ======================================================

def get_analysis(kind, ticker):
    if kind == "Basic Info":
        return get_basic_stock_info(ticker)
    if kind == "Technical Analysis":
        return get_technical_analysis(ticker)
    if kind == "Financial Analysis":
        return financial_analysis(ticker)
    if kind == "Filings Analysis":
        return filings_analysis(ticker)
    if kind == "News Analysis":
        return news_summary(ticker)
    if kind == "Recommend":
        return recommend(ticker)
    if kind == "Sentiment Analysis":
        return get_sentiment(ticker)
    if kind == "Research Analysis":
        return research(company_stock=ticker)
    if kind == "Real-Time Price":
        price = get_latest_price(ticker)
        return f"Live Price: {price}"
    return "Unknown"


# ======================================================
# UI Result
# ======================================================

st.title("AI Stock Analyst — Phase 1")

if st.sidebar.button("Run Analysis"):
    if not ticker:
        st.error("Ticker required")
    else:
        st.write(f"### Analysis: {analysis_type} for {ticker}")
        result = get_analysis(analysis_type, ticker)
        st.write(result)

        predicted = predict_next_price(ticker, timeframe_mapping[timeframe])
        if predicted:
            st.write(f"📈 Predicted Next Price: **${predicted:.2f}**")

        last_price = get_latest_price(ticker)
        if last_price:
            st.metric("Latest Price", f"${last_price:.2f}")

st.write("---")
st.write("💡 Phase 2: Multi-Tab Dashboard + WebSocket Live Charts Coming Next")
