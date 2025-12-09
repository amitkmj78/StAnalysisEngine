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

# WebSocket client (optional real-time feed)

FINNHUB_WS = f"wss://ws.finnhub.io?token={os.getenv('FINNHUB_API_KEY')}"

# Global state to hold latest price
latest_price = st.session_state.get("latest_price", None)

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

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None

# Importing agent-like helper functions (your modules)
from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research

# ------------------------------------------------------------------
#                    ENVIRONMENT SETUP & VALIDATION
# ------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WS_PRICE_FEED_URL = os.getenv("WS_PRICE_FEED_URL")  # optional WebSocket URL



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

st.info(f"Today's Date: {datetime.date.today()}")

# ------------------------------------------------------------------
#                          LLM SETUP
# ------------------------------------------------------------------

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

if not available_llms:
    st.error("❌ No usable LLM instance could be created. Double-check your API keys.")
    st.stop()

select_llm = st.sidebar.selectbox("Select LLM Type", available_llms, key="llm_select")

if select_llm == "Groq Llama3-70B":
    llm = llm_groq
else:
    llm = llm_openai
# ------------------------------------------------------------------
#                         SIDEBAR / UI INPUTS
# ------------------------------------------------------------------

st.sidebar.header("Stock Query")
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Stream Live Price"):
    start_price_feed(ticker)
    st.success(f"Real-time feed started for {ticker}")

placeholder = st.empty()

while "latest_price" in st.session_state:
    price = st.session_state.latest_price
    if price:
        placeholder.metric(label=f"Live Price: {ticker.upper()}", value=f"${price:.2f}")
    time.sleep(1)
raw_query = st.sidebar.text_input("Enter Stock ticker (e.g., AAPL):", value="AAPL", key="ticker_input")
query = raw_query.strip().upper()

prediction_days = st.sidebar.number_input(
    "Prediction Days",
    min_value=1,
    max_value=365,
    value=5,
    step=1,
    help="Select the number of future trading days to predict.",
    key="pred_days",
)

include_earnings_impact = st.sidebar.checkbox(
    "Include Earnings Impact using AI (LLM)",
    value=True,
    key="earnings_checkbox",
)

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "Research Analysis",
        "Basic Info",
        "Technical Analysis",
        "Financial Analysis",
        "Filings Analysis",
        "News Analysis",
        "Recommend",
        "Real-Time Price",
        "Sentiment Analysis",
    ],
    key="analysis_type",
)

timeframe_mapping = {
    "1 Year": "1y",
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Month": "6mo",
    "5 Year": "5y",
}

timeframe = st.sidebar.radio(
    "Select Timeframe:",
    list(timeframe_mapping.keys()),
    key="timeframe",
)

chart_type = st.sidebar.selectbox(
    "Select Chart Type:",
    ["Line Chart", "Candlestick Chart"],
    key="chart_type",
)

include_sentiment = st.sidebar.checkbox(
    "Include Sentiment Adjustment for Predicted Price",
    value=False,
    key="sentiment_checkbox",
)

ml_model_type = st.sidebar.selectbox(
    "ML Model for Price Prediction",
    ["Single Gradient Boosting", "Ensemble Gradient Boosting"],
    key="ml_model_type",
)

enable_ws_live_feed = st.sidebar.checkbox(
    "Enable Real-time WebSocket Price Feed (if configured)",
    value=False,
    key="ws_live_feed",
)

research_prompt = st.sidebar.text_input(
    "Modify Research Prompt (Optional)",
    placeholder="Enter custom research prompt for the Research analysis...",
    key="research_prompt_input",
)

analyze_button = st.sidebar.button("📊 Analyze", key="analyze_button")

# ------------------------------------------------------------------
#                         DATA FETCHING (CACHED)
# ------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_price_chart(ticker: str, period: str) -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame()
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval="1d").dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def get_real_time_stock_price(ticker: str):
    if not ticker:
        return "N/A"
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d").dropna()
        return float(data["Close"].iloc[-1]) if not data.empty else "N/A"
    except Exception as e:
        st.error(f"Error fetching real-time price for {ticker}: {e}")
        return "N/A"


@st.cache_data(ttl=60, show_spinner=False)
def get_latest_stock_price(ticker: str):
    if not ticker:
        return "N/A"
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d").dropna()
        return round(float(data["Close"].iloc[-1]), 2) if not data.empty else "N/A"
    except Exception as e:
        st.error(f"Error fetching latest stock price for {ticker}: {e}")
        return "N/A"

# ------------------------------------------------------------------
#                      ADVANCED ML PREDICTION LOGIC
# ------------------------------------------------------------------

def train_gbm_model(X, y, n_estimators=300, learning_rate=0.01, max_depth=4):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X, y)
    return model


def ensemble_predict(models, X_new: np.ndarray) -> float:
    preds = [m.predict(X_new)[0] for m in models]
    return float(np.mean(preds))


def predict_next_stock_price(ticker: str, period: str):
    """Predict next-day price using advanced GBM / ensemble."""
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 20:
        return None

    df = data.copy()
    df["Index"] = np.arange(len(df))
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Lag1"] = df["Close"].shift(1)
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()

    if df.empty:
        return None

    features = ["Index", "Lag1", "MA5", "MA10", "Return"]
    X = df[features].values
    y = df["Close"].values

    model_1 = train_gbm_model(X, y, n_estimators=300, learning_rate=0.01, max_depth=4)
    models = [model_1]

    if ml_model_type == "Ensemble Gradient Boosting":
        model_2 = train_gbm_model(X, y, n_estimators=500, learning_rate=0.015, max_depth=3)
        models.append(model_2)

    last_row = df.iloc[-1]
    next_index = last_row["Index"] + 1
    X_new = np.array([[next_index, last_row["Close"], last_row["MA5"], last_row["MA10"], last_row["Return"]]])

    if len(models) == 1:
        predicted = float(models[0].predict(X_new)[0])
    else:
        predicted = ensemble_predict(models, X_new)

    return predicted


def predict_next_30_days(ticker: str, period: str, days_ahead: int = 30):
    """Predict next N trading days with feature recursion & optional earnings impact."""
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 40:
        return None, None

    df = data.copy()
    df["Index"] = np.arange(len(df))
    df["Lag1"] = df["Close"].shift(1)
    df["MA3"] = df["Close"].rolling(window=3).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Close"].rolling(window=10).std()
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df = df.dropna()

    if df.empty:
        return None, None

    features = ["Index", "Lag1", "MA3", "MA10", "Return", "Volatility", "Gap"]
    X = df[features].values
    y = df["Close"].values

    base_model = train_gbm_model(X, y, n_estimators=500, learning_rate=0.01, max_depth=4)
    models = [base_model]

    if ml_model_type == "Ensemble Gradient Boosting":
        model_ens = train_gbm_model(X, y, n_estimators=700, learning_rate=0.008, max_depth=3)
        models.append(model_ens)

    last_index = df["Index"].iloc[-1]
    last_close = df["Close"].iloc[-1]
    last_MA3 = df["MA3"].iloc[-1]
    last_MA10 = df["MA10"].iloc[-1]
    last_return = df["Return"].iloc[-1]
    last_vol = df["Volatility"].iloc[-1]
    last_gap = df["Gap"].iloc[-1]

    predictions = []

    for i in range(1, days_ahead + 1):
        new_index = last_index + i
        X_new = np.array([[new_index, last_close, last_MA3, last_MA10, last_return, last_vol, last_gap]])

        if len(models) == 1:
            pred = float(models[0].predict(X_new)[0])
        else:
            pred = ensemble_predict(models, X_new)

        predictions.append(pred)

        new_return = (pred - last_close) / last_close if last_close != 0 else 0
        new_MA3 = (2 / 3) * last_MA3 + (1 / 3) * pred
        new_MA10 = (9 / 10) * last_MA10 + (1 / 10) * pred
        new_vol = 0.9 * last_vol + 0.1 * abs(new_return)
        new_gap = last_gap

        last_close = pred
        last_return = new_return
        last_MA3 = new_MA3
        last_MA10 = new_MA10
        last_vol = new_vol
        last_gap = new_gap

    if include_earnings_impact:
        earnings_impact = get_earnings_impact(ticker)
        factor_earnings = 0.05
        st.info(f"AI Earnings Impact factor for {ticker}: {earnings_impact}")
        predictions = [p * (1 + factor_earnings * earnings_impact) for p in predictions]

    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=days_ahead)

    prediction_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
    prediction_df["Date"] = prediction_df["Date"].astype(str)
    st.dataframe(prediction_df)

    return future_dates, predictions

# ------------------------------------------------------------------
#                    SENTIMENT & RESEARCH FUNCTIONS
# ------------------------------------------------------------------

def get_earnings_impact(ticker: str) -> float:
    """
    Use the LLM directly to get an earnings sentiment score between -1 and 1.
    """
    if llm is None:
        return 0.0

    prompt = (
        f"Analyze the latest quarterly earnings report for {ticker}. "
        "Based on revenue growth, profit margins, forward guidance, and overall fundamentals, "
        "provide a sentiment score on a scale from -1 (very negative) to 1 (very positive). "
        "Return ONLY the numeric score, nothing else."
    )

    try:
        result = llm.invoke(prompt)
        text = getattr(result, "content", str(result))
    except Exception as e:
        st.error(f"Error calling LLM for earnings impact: {e}")
        return 0.0

    match = re.search(r"(-?\d+\.?\d*)", text)
    if match:
        try:
            impact = float(match.group(1))
            return max(min(impact, 1.0), -1.0)
        except Exception as e:
            st.error(f"Error parsing numeric earnings impact: {e}")
            return 0.0
    else:
        st.warning("No numeric score found in LLM earnings impact output.")
        return 0.0


def get_sentiment_analysis(ticker: str) -> str:
    if not TAVILY_API_KEY:
        return "Sentiment analysis unavailable: TAVILY_API_KEY not configured."

    tavily = TavilySearchResults(api_key=TAVILY_API_KEY)
    try:
        search_results = tavily.run(
            f"{ticker} stock market news sentiment analysis, recent headlines, and outlook"
        )
        return f"News & Sentiment Context for {ticker}:\n\n{search_results}"
    except Exception as e:
        return f"Error during Tavily sentiment search: {e}"


def adjust_prediction_with_sentiment(prediction: float, ticker: str):
    sentiment_text = get_sentiment_analysis(ticker)

    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk

        nltk.download("vader_lexicon", quiet=True)
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(sentiment_text)
        compound = scores["compound"]
    except Exception as e:
        st.warning(f"Could not compute VADER sentiment, skipping adjustment: {e}")
        return prediction, {"compound": 0.0}

    factor = 0.02
    adjusted = prediction * (1 + factor * compound)
    return adjusted, scores

# ------------------------------------------------------------------
#                  SIMPLE ANALYSIS DISPATCHER (NO AGENTS)
# ------------------------------------------------------------------

def get_analysis(analysis_type: str, ticker: str):
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

    if analysis_type == "Real-Time Price":
        price = get_real_time_stock_price(ticker)
        return f"Real-time price for {ticker}: {price}"

    if analysis_type == "Sentiment Analysis":
        return get_sentiment_analysis(ticker)

    if analysis_type == "Research Analysis":
        return research(company_stock=ticker, user_prompt=research_prompt)

    return f"Unknown analysis type: {analysis_type}"

# ------------------------------------------------------------------
#                REAL-TIME WEBSOCKET PRICE FEED (OPTIONAL)
# ------------------------------------------------------------------

def websocket_price_feed(symbol: str):
    """
    Example WebSocket price feed listener.
    Requires WS_PRICE_FEED_URL pointing to a provider that sends JSON
    with fields like: {"symbol": "AAPL", "price": 195.34}
    """
    if websocket is None:
        st.warning("websocket-client is not installed. Run `pip install websocket-client` to use WebSocket feed.")
        return

    if not WS_PRICE_FEED_URL:
        st.warning("WS_PRICE_FEED_URL is not configured in environment. WebSocket feed disabled.")
        return

    def on_message(ws, message):
        try:
            import json
            data = json.loads(message)
            if data.get("symbol", "").upper() == symbol.upper():
                st.session_state.live_price = float(data.get("price"))
        except Exception:
            pass

    def on_error(ws, err):
        print("WebSocket error:", err)

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket closed:", close_status_code, close_msg)

    def on_open(ws):
        # If provider requires subscription message, send it here.
        pass

    ws = websocket.WebSocketApp(
        WS_PRICE_FEED_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()

# ------------------------------------------------------------------
#                         DATA & VISUALIZATION
# ------------------------------------------------------------------

def ShowData():
    if not query:
        st.error("Please enter a valid stock ticker.")
        return

    stock_data = get_stock_price_chart(query, timeframe_mapping[timeframe])
    if stock_data.empty:
        st.warning(f"No data available for {query} for the selected timeframe.")
        return

    current_price = get_latest_stock_price(query)
    predicted_price = predict_next_stock_price(query, timeframe_mapping[timeframe])

    stock_data["Smoothed"] = stock_data["Close"].rolling(window=5, min_periods=1).mean()

    # --- Graph 1: Historical Price Trend with Prediction ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data["Smoothed"],
        mode="lines",
        line=dict(color="darkgreen", width=3),
        name="Smoothed Closing Price",
    ))
    fig1.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data["Close"],
        mode="lines+markers",
        line=dict(color="#02D285", width=2),
        marker=dict(size=6, symbol="circle-open"),
        name="Closing Price",
        text=[f"${val:.2f}" for val in stock_data["Close"]],
        textposition="top center",
        hovertemplate="%{text}",
    ))

    if isinstance(current_price, (int, float, np.floating)):
        latest_date = stock_data.index[-1]
        fig1.add_trace(go.Scatter(
            x=[latest_date],
            y=[current_price],
            mode="markers+text",
            marker=dict(color="#facff8", size=12, line=dict(color="black", width=1)),
            text=[f"${current_price:.2f}"],
            textposition="top right",
            name=f"Last Day Price: ${current_price}",
        ))

    sentiment_str = ""
    display_pred_price = predicted_price

    if predicted_price is not None:
        if len(stock_data.index) > 1:
            delta = stock_data.index[-1] - stock_data.index[-2]
            next_date = stock_data.index[-1] + delta
        else:
            next_date = stock_data.index[-1]

        if include_sentiment:
            display_pred_price, sentiment_scores = adjust_prediction_with_sentiment(predicted_price, query)
            sentiment_str = f"<br>AI News Sentiment (compound): {sentiment_scores['compound']:.2f}"
        else:
            display_pred_price = predicted_price

        fig1.add_shape(
            type="line",
            x0=next_date,
            x1=next_date,
            y0=min(stock_data["Close"].min(), display_pred_price) * 0.95,
            y1=max(stock_data["Close"].max(), display_pred_price) * 1.05,
            line=dict(color="purple", dash="dash"),
        )
        fig1.add_annotation(
            x=next_date,
            y=display_pred_price,
            text=f"Predicted: ${display_pred_price:.2f}{sentiment_str}<br>{next_date.strftime('%Y-%m-%d')}",
            showarrow=True,
            arrowhead=2,
            ax=10,
            ay=-40,
            font=dict(color="purple"),
        )

    fig1.update_layout(
        title=f"{query} Stock Price Trend ({timeframe})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        xaxis=dict(tickformat="%b %d", tickangle=45),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Graph 2: Bar Chart Comparison for Overall Price ---
    if predicted_price is not None and isinstance(current_price, (int, float, np.floating)):
        if include_sentiment:
            display_price, _ = adjust_prediction_with_sentiment(predicted_price, query)
        else:
            display_price = predicted_price

        fig2 = go.Figure(data=[
            go.Bar(name="Last Day Price", x=["Last Day Price"], y=[current_price], marker_color="#5DE2E7"),
            go.Bar(name="AI Predicted Next Price", x=["Predicted Next Price"], y=[display_price], marker_color="#4AFC5C"),
        ])
        fig2.update_layout(
            title=f"{query} Price Comparison",
            yaxis_title="Price (USD)",
            barmode="group",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.metric(label=f"Last Day Price of {query}", value=f"${current_price:.2f}")
        st.metric(label="AI Predicted Next Price", value=f"${display_price:.2f}")
    else:
        st.warning("Not enough data to predict the next stock price.")

    # --- Graph 3: Last 30 Days Actual vs Predicted (backtest) ---
    if len(stock_data) >= 30:
        actual_prices = []
        predicted_prices = []
        date_labels = []

        n = len(stock_data)

        for i in range(n - 30, n):
            if i < 5:
                continue

            sub_data = stock_data.iloc[:i].copy()
            sub_data["Index"] = np.arange(len(sub_data))
            sub_data["Lag1"] = sub_data["Close"].shift(1)
            sub_data["MA3"] = sub_data["Close"].rolling(window=3).mean()
            sub_data["Return"] = sub_data["Close"].pct_change()
            sub_data = sub_data.dropna()
            if len(sub_data) < 5:
                continue

            features = ["Index", "Lag1", "MA3", "Return"]
            X = sub_data[features].values
            y = sub_data["Close"].values

            temp_model = train_gbm_model(X, y, n_estimators=150, learning_rate=0.02, max_depth=3)
            last_index_sub = sub_data["Index"].iloc[-1] + 1
            last_lag1 = sub_data["Close"].iloc[-1]
            last_ma3 = sub_data["MA3"].iloc[-1]
            last_return = sub_data["Return"].iloc[-1]
            X_new = np.array([[last_index_sub, last_lag1, last_ma3, last_return]])
            pred = float(temp_model.predict(X_new)[0])

            predicted_prices.append(pred)
            actual_prices.append(stock_data["Close"].iloc[i])
            date_labels.append(stock_data.index[i].strftime("%b %d"))

        if date_labels:
            fig3 = go.Figure(data=[
                go.Bar(name="Actual Price", x=date_labels, y=actual_prices, marker_color="#5DE2E7"),
                go.Bar(name="Predicted Price", x=date_labels, y=predicted_prices, marker_color="#4AFC5C"),
            ])
            fig3.update_layout(
                title=f"{query} Last 30 Days: Actual vs Predicted Prices",
                yaxis_title="Price (USD)",
                barmode="group",
                template="plotly_white",
            )
            st.plotly_chart(fig3, use_container_width=True)

            # --- Graph 4: Average of Last 30 Days Actual vs Predicted ---
            avg_actual = float(np.mean(actual_prices)) if actual_prices else 0
            avg_predicted = float(np.mean(predicted_prices)) if predicted_prices else 0

            fig4 = go.Figure(data=[
                go.Bar(name="Avg Actual", x=["Avg Actual"], y=[avg_actual], marker_color="#5DE2E7"),
                go.Bar(name="Avg Predicted", x=["Avg Predicted"], y=[avg_predicted], marker_color="#4AFC5C"),
            ])
            fig4.update_layout(
                title="Average of Last 30 Days: Actual vs Predicted",
                yaxis_title="Price (USD)",
                barmode="group",
                template="plotly_white",
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.metric(label="Avg Actual Price (Last 30 Days)", value=f"${avg_actual:.2f}")
            st.metric(label="Avg Predicted Price (Last 30 Days)", value=f"${avg_predicted:.2f}")
        else:
            st.warning("Not enough data to generate last 30 days predictions comparison.")

        # --- Graph 5: Predicted Next N Days Prices ---
        future_dates, future_predictions = predict_next_30_days(
            query,
            timeframe_mapping[timeframe],
            days_ahead=prediction_days,
        )
        if future_dates is not None and future_predictions is not None:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode="lines+markers",
                marker=dict(size=8),
                line=dict(width=2),
                name="Predicted Price",
            ))
            fig5.update_layout(
                title=f"{query} GEN AI Agent Based Predicted Prices for Next {prediction_days} Days",
                xaxis_title="Date",
                yaxis_title="Predicted Price (USD)",
                template="plotly_white",
                xaxis=dict(tickangle=45),
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Not enough data to predict the next days' prices.")
    else:
        st.warning("Not enough historical data to run the last 30 days backtest.")

# ------------------------------------------------------------------
#                           MAIN EXECUTION
# ------------------------------------------------------------------

if "responses" not in st.session_state:
    st.session_state.responses = []

if "live_price" not in st.session_state:
    st.session_state.live_price = None

# Optional WebSocket live feed
if enable_ws_live_feed and query:
    websocket_price_feed(query)
    live_price_placeholder = st.empty()
    if st.session_state.live_price is not None:
        live_price_placeholder.metric(
            label=f"Live WebSocket Price for {query}",
            value=f"${st.session_state.live_price:.4f}",
        )
    else:
        live_price_placeholder.info(
            "Waiting for WebSocket live price updates... "
            "(configure WS_PRICE_FEED_URL and provider subscription)."
        )

if analyze_button:
    if not query:
        st.error("Please enter a valid stock ticker.")
    else:
        ShowData()
        st.info(f"Analyzing stock: {query}")
        try:
            if analysis_type == "Recommend":
                basic_info = get_analysis("Basic Info", query)
                financial_info = get_analysis("Financial Analysis", query)
                sentiment_info = get_analysis("Sentiment Analysis", query)
                recommendation = get_analysis("Recommend", query)

                response = (
                    f"Basic Info:\n{basic_info}\n\n"
                    f"Financial Analysis:\n{financial_info}\n\n"
                    f"Sentiment Analysis:\n{sentiment_info}\n\n"
                    f"Final Recommendation:\n{recommendation}"
                )
            else:
                response = get_analysis(analysis_type, query)

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
            st.error(f"An error occurred while running analysis: {e}")

st.markdown("### Analysis History")
if st.session_state.responses:
    for entry in st.session_state.responses:
        st.markdown(f"**Query:** {entry.get('query', 'N/A')}")
        st.markdown(f"**Analysis Type:** {entry.get('analysis_type', 'N/A')}")
        st.markdown(f"**Response:**\n{entry.get('response', 'No response')}")
        st.markdown("---")
else:
    st.write("No analysis has been run yet.")
