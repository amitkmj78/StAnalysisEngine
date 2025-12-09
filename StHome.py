import os
import datetime
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Import your agent/helper functions (these can stay as-is) ---
from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend  # if you want to keep using it
from Agent.reasearchAgent import research

# ==========================
# ENV & MODEL SETUP
# ==========================
load_dotenv()

st.info(f"Today's Date: {datetime.date.today()}")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LLMs
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
llm_groq = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY) if GROQ_API_KEY else None

select_llm = st.sidebar.selectbox("Select LLM Type", ["Open AI", "Ollama-Local"])
llm = llm_openai if select_llm == "Open AI" else llm_groq

# ==========================
# STOCK DATA FUNCTIONS
# ==========================

def get_real_time_stock_price(ticker: str):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d").dropna()
    return data["Close"].iloc[-1] if not data.empty else "N/A"


def get_stock_price_chart(ticker: str, period: str):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval="1d").dropna()
    return data


def get_latest_stock_price(ticker: str):
    data = get_stock_price_chart(ticker, "1d")
    return round(data["Close"].iloc[-1], 2) if not data.empty else "N/A"


def predict_next_stock_price(ticker: str, period: str):
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 2:
        return None

    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    next_index = np.array([[len(data)]])
    predicted = model.predict(next_index)[0]
    return predicted


def predict_next_30_days(ticker: str, period: str, days_ahead: int = 30):
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 2:
        return None, None

    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X, y)

    future_indices = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
    predictions = model.predict(future_indices)

    last_date = data.index[-1]
    if len(data.index) > 1:
        delta = data.index[-1] - data.index[-2]
    else:
        delta = datetime.timedelta(days=1)

    future_dates = [last_date + (i + 1) * delta for i in range(days_ahead)]
    return future_dates, predictions


# ==========================
# SIDEBAR UI
# ==========================

timeframe_mapping = {
    "1 Day": "1d",
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Month": "6mo",
    "1 Year": "1y",
    "5 Year": "5y",
}

st.sidebar.header("Stock Query")
query = st.sidebar.text_area("Enter Stock ticker (e.g., AAPL):", value="AAPL")

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "Basic Info",
        "Technical Analysis",
        "Financial Analysis",
        "Filings Analysis",
        "Research Analysis",
        "News Analysis",
        "Recommend",
        "Real-Time Price",
        "Sentiment Analysis",
    ],
)

timeframe = st.sidebar.radio("Select Timeframe:", list(timeframe_mapping.keys()))
chart_type = st.sidebar.selectbox("Select Chart Type:", ["Line Chart", "Candlestick Chart"])

research_prompt = st.sidebar.text_input(
    label="Modify Research Prompt (Optional)",
    placeholder="Enter custom prompt...",
    key="research_prompt",
)

analyze_button = st.sidebar.button("📊 Analyze")


# ==========================
# SENTIMENT / SEARCH HELPERS
# ==========================

def get_sentiment_analysis(ticker: str):
    tv = TavilySearchResults(api_key=TAVILY_API_KEY)
    result = tv.run(f"{ticker} stock market news sentiment analysis")
    return f"Sentiment Analysis for {ticker}:\n\n{result}"


# ==========================
# ANALYSIS DISPATCHER (NO AGENTS)
# ==========================

def run_analysis(analysis_type: str, ticker: str):
    """Call the correct function based on analysis_type."""
    if analysis_type == "Basic Info":
        return get_basic_stock_info(ticker)

    if analysis_type == "Technical Analysis":
        return get_technical_analysis(ticker)

    if analysis_type == "Financial Analysis":
        return financial_analysis(ticker)

    if analysis_type == "Filings Analysis":
        return filings_analysis(ticker)

    if analysis_type == "Research Analysis":
        # Your research function expects company_stock + user_prompt
        return research(company_stock=ticker, user_prompt=research_prompt)

    if analysis_type == "News Analysis":
        return news_summary(ticker)

    if analysis_type == "Real-Time Price":
        price = get_real_time_stock_price(ticker)
        return f"Real-time price for {ticker}: {price}"

    if analysis_type == "Sentiment Analysis":
        return get_sentiment_analysis(ticker)

    return "Unknown analysis type."


# ==========================
# CHARTING / VISUALIZATION
# ==========================

def ShowData():
    stock_data = get_stock_price_chart(query, timeframe_mapping[timeframe])
    current_price = get_latest_stock_price(query)
    predicted_price = predict_next_stock_price(query, timeframe_mapping[timeframe])

    if stock_data.empty:
        st.warning(f"No data available for {query} for the selected timeframe.")
        return

    stock_data["Smoothed"] = stock_data["Close"].rolling(window=5, min_periods=1).mean()

    # Graph 1: Historical Price Trend with Prediction
    fig1, ax1 = plt.subplots(figsize=(18, 12))
    ax1.plot(
        stock_data.index,
        stock_data["Smoothed"],
        label="Smoothed Closing Price",
        linewidth=3,
        color="darkgreen",
        linestyle="-",
    )
    ax1.plot(
        stock_data.index,
        stock_data["Close"],
        label="Closing Price",
        linewidth=2,
        color="green",
        marker="o",
        markersize=6,
        alpha=0.8,
        linestyle="-",
        markerfacecolor="white",
    )
    ax1.scatter(stock_data.index, stock_data["Close"], color="#81c784", s=50, alpha=0.7, zorder=3)

    latest_date = stock_data.index[-1]
    ax1.scatter(
        latest_date,
        current_price,
        color="red",
        s=120,
        label=f"Last Day Price: ${current_price}",
        edgecolors="black",
        zorder=4,
    )

    if isinstance(current_price, (int, float, np.floating)):
        ax1.annotate(
            f"${current_price:.2f}",
            (latest_date, current_price),
            textcoords="offset points",
            xytext=(10, 10),
            ha="left",
            fontsize=12,
            color="red",
            fontweight="bold",
        )

    if predicted_price is not None:
        if len(stock_data.index) > 1:
            delta = stock_data.index[-1] - stock_data.index[-2]
            next_date = stock_data.index[-1] + delta
        else:
            next_date = stock_data.index[-1]

        ax1.axvline(x=next_date, color="purple", linestyle="--", label="Predicted Next Price")
        ax1.annotate(
            f"Predicted: ${predicted_price:.2f}\non {next_date.strftime('%Y-%m-%d')}",
            (next_date, predicted_price),
            textcoords="offset points",
            xytext=(10, -10),
            ha="left",
            fontsize=12,
            color="purple",
            fontweight="bold",
        )

    ax1.set_facecolor("#f4f4f4")
    ax1.set_title(f"{query} Stock Price Trend ({timeframe})", fontsize=18, fontweight="bold", color="#333")
    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Price (USD)", fontsize=14)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend()
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    st.pyplot(fig1)

    # Simple bar chart: current vs predicted
    if predicted_price is not None and isinstance(current_price, (int, float, np.floating)):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        prices = [current_price, predicted_price]
        labels = ["Last Day Price", "Predicted Next Price"]
        ax2.bar(labels, prices, alpha=0.8)
        for i, price in enumerate(prices):
            ax2.text(i, price, f"${price:.2f}", ha="center", va="bottom", fontsize=12)
        ax2.set_ylabel("Price (USD)")
        ax2.set_title(f"{query} Price Comparison")
        st.pyplot(fig2)

        st.metric(label=f"Last Day Price of {query}", value=f"${current_price}")
        st.metric(label="Predicted Next Price", value=f"${predicted_price:.2f}")

    # Next 10 days prediction
    future_dates, future_predictions = predict_next_30_days(
        query, timeframe_mapping[timeframe], days_ahead=10
    )
    if future_dates is not None and future_predictions is not None:
        fig5, ax5 = plt.subplots(figsize=(18, 8))
        ax5.plot(future_dates, future_predictions, marker="o", linestyle="-", label="Predicted Price")
        ax5.set_title(f"{query} Predicted Prices for Next 10 Days", fontsize=18, fontweight="bold")
        ax5.set_xlabel("Date", fontsize=14)
        ax5.set_ylabel("Predicted Price (USD)", fontsize=14)
        ax5.tick_params(axis="x", rotation=45)
        ax5.legend()
        st.pyplot(fig5)
    else:
        st.warning("Not enough data to predict the next 10 days' prices.")


# ==========================
# MAIN EXECUTION
# ==========================

if "responses" not in st.session_state:
    st.session_state.responses = []

if analyze_button:
    ShowData()
    st.info(f"Analyzing stock: {query}")

    try:
        if analysis_type == "Recommend":
            basic_info = run_analysis("Basic Info", query)
            financial_info = run_analysis("Financial Analysis", query)
            sentiment_info = run_analysis("Sentiment Analysis", query)

            if llm is None:
                response = (
                    "No LLM API key configured.\n\n"
                    f"Basic Info:\n{basic_info}\n\n"
                    f"Financial Analysis:\n{financial_info}\n\n"
                    f"Sentiment:\n{sentiment_info}\n"
                )
            else:
                prompt = f"""
You are an investment assistant. Based on the following information, give a clear, concise
investment recommendation (buy/hold/sell) for the stock {query}, with reasoning.

Basic Info:
{basic_info}

Financial Analysis:
{financial_info}

Sentiment Analysis:
{sentiment_info}

Respond with a short summary, a recommendation, and key risks.
"""
                llm_result = llm.invoke(prompt)
                response_text = getattr(llm_result, "content", str(llm_result))
                response = response_text
        else:
            response = run_analysis(analysis_type, query)

        st.success(f"{analysis_type} Complete")
        st.write(response)

        st.session_state.responses.append(
            {
                "query": query,
                "analysis_type": analysis_type,
                "response": response,
            }
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("### Analysis History")
if st.session_state.responses:
    for entry in st.session_state.responses:
        st.markdown(f"**Query:** {entry.get('query', 'N/A')}")
        st.markdown(f"**Analysis Type:** {entry.get('analysis_type', 'N/A')}")
        st.markdown(f"**Response:**\n{entry.get('response', 'No response')}")
        st.markdown("---")
else:
    st.write("No analysis has been run yet.")
