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
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Importing agent functions (adjust paths as needed)
from Agent.newAgent import news_summary
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research

# Load environment variables
load_dotenv()

# Set up conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Display today's date
st.info(f"Today's Date: {datetime.date.today()}")

# API Key setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize LLMs
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.7, openai_api_key=OPENAI_API_KEY)
llm_ollama = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# Select LLM Model from sidebar
select_llm = st.sidebar.selectbox("Select LLM Type", ["Ollama-Local", "Open AI"])
llm = llm_openai if select_llm == "Open AI" else llm_ollama

# Data fetching functions
def get_real_time_stock_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d").dropna()
    return data["Close"].iloc[-1] if not data.empty else "N/A"

def get_stock_price_chart(ticker, period):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval="1d").dropna()
    return data

def get_latest_stock_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d").dropna()
    return round(data["Close"].iloc[-1], 2) if not data.empty else "N/A"

# Prediction function using Gradient Boosting Regressor.
def predict_next_stock_price(ticker, period):
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 2:
        return None
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X, y)
    next_index = np.array([[len(data)]])
    predicted = model.predict(next_index)[0]
    return predicted

# New Prediction Function: Predict the next 30 days of stock prices.
# def predict_next_30_days(ticker, period, days_ahead=30):
#     data = get_stock_price_chart(ticker, period)
#     st.write(data)
#     if data.empty or len(data) < 2:
#         return None, None
#     X = np.arange(len(data)).reshape(-1, 1)
#     y = data["Close"].values
#     model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
#     model.fit(X, y)
#     future_indices = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
#     predictions = model.predict(future_indices)
#     last_date = data.index[-1]
#     if len(data.index) > 1:
#         delta = data.index[-1] - data.index[-2]
#     else:
#         delta = datetime.timedelta(days=1)
#     future_dates = [last_date + (i+1)*delta for i in range(days_ahead)]
#     st.write(predictions)
#     return future_dates, predictions

# def predict_next_30_days(ticker, period, days_ahead=30):
#     data = get_stock_price_chart(ticker, period)
#     st.write("Historical Data:", data)
#     if data.empty or len(data) < 2:
#         return None, None
#     # Create a simple sequential index (assumes equally spaced trading days)
#     X = np.arange(len(data)).reshape(-1, 1)
#     y = data["Close"].values

#     # Fit the model using Gradient Boosting Regressor.
#     model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, 
#                                         max_depth=3, random_state=42)
#     model.fit(X, y)
    
#     # Prepare future indices for prediction.
#     future_indices = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
#     predictions = model.predict(future_indices)
    
#     # Instead of using the delta between the last two dates, generate the next trading dates.
#     # Here we use a business day frequency.
#     last_date = data.index[-1]
#     # The next date will be one business day after the last_date.
#     future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=days_ahead)
    
#     st.write("Predicted Prices:", predictions)
#     return future_dates, predictions


def predict_next_5_days(ticker, period, days_ahead=5):

    data = get_stock_price_chart(ticker, period)
    st.write("Historical Data:", data)
    if data.empty or len(data) < 2:
        return None, None

    # Create a DataFrame with a sequential index and a lag of 1 day.
    df = data.copy()
    df["Index"] = np.arange(len(df))
    df["Lag1"] = df["Close"].shift(1)
    df = df.dropna()  # Drop the first row with NaN lag value

    # Features: [Sequential Index, Lag1]
    X = df[["Index", "Lag1"]].values
    y = df["Close"].values

    # Fit the model using Gradient Boosting Regressor.
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X, y)

    # Retrieve the last known index and closing price.
    last_index = df["Index"].iloc[-1]
    last_close = df["Close"].iloc[-1]

    predictions = []
    # Iteratively predict the next day and update the lag feature.
    for i in range(1, days_ahead + 1):
        new_index = last_index + i
        # Use the current last_close (initially from the data, then updated with each prediction)
        X_new = np.array([[new_index, last_close]])
        pred = model.predict(X_new)[0]
        predictions.append(pred)
        # Update the lag for the next day's prediction.
        last_close = pred

    # Generate future trading dates using business day frequency.
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=days_ahead)
    
    #st.write("Predicted Prices:", predictions)
    return future_dates, predictions





# Sentiment analysis functions.
# Perform sentiment analysis using Tavily Search.
def get_sentiment_analysis(ticker):
    search_results = TavilySearchResults(api_key=TAVILY_API_KEY).run(f"{ticker} stock market news sentiment analysis")
    return f"Sentiment Analysis for {ticker}:\n\n{search_results}"

# Adjust the predicted price based on sentiment.
def adjust_prediction_with_sentiment(prediction, ticker):
    # Get sentiment analysis text.
    sentiment_text = get_sentiment_analysis(ticker)
    # Import and download the VADER lexicon if necessary.
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentiment_text)
    compound = scores['compound']
    factor = 0.02  # Adjust factor as needed.
    adjusted = prediction * (1 + factor * compound)
    return adjusted, scores

# Updated timeframe mapping options.
timeframe_mapping = {
    "1 Day": "1d",
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Month": "6mo",
    "1 Year": "1y",
    "5 Year": "5y"
}

# Sidebar inputs.
st.sidebar.header("Stock Query")
query = st.sidebar.text_area(label="Enter Stock ticker (e.g., AAPL):", value="AAPL")
analysis_type = st.sidebar.selectbox("Select Analysis Type", 
    ["Basic Info", "Technical Analysis", "Financial Analysis", "Filings Analysis", 
     "Research Analysis", "News Analysis", "Recommend", "Real-Time Price", "Sentiment Analysis"])
timeframe = st.sidebar.radio("Select Timeframe:", list(timeframe_mapping.keys()))
chart_type = st.sidebar.selectbox("Select Chart Type:", ["Line Chart", "Candlestick Chart"])
# New checkbox to include sentiment adjustment.
include_sentiment = st.sidebar.checkbox("Include Sentiment Adjustment for Predicted Price", value=False)
analyze_button = st.sidebar.button("📊Analyze")  # Using an icon button

# Create a research prompt input once (with a unique key)
research_prompt = st.sidebar.text_input(
    label="Modify Prompt (Optional)",
    placeholder="Enter custom prompt...",
    key="research_prompt"
)

# Define tools.
tools = {
    "News Summary": Tool(
        name="News Summary",
        func=news_summary,
        description="Fetches and summarizes the latest news for a given query."
    ),
    "Tavily Search": Tool(
        name="Tavily Search",
        func=lambda q: TavilySearchResults(api_key=TAVILY_API_KEY).run(q),
        description="Fetches real-time web search results using Tavily."
    ),
    "Basic Stock Info": Tool(
        name="Basic Stock Info",
        func=get_basic_stock_info,
        description="Fetch basic stock information for a given ticker symbol."
    ),
    "Technical Analysis": Tool(
        name="Technical Analysis",
        func=get_technical_analysis,
        description="Perform technical analysis for a given ticker symbol."
    ),
    "Financial Analysis": Tool(
        name="Financial Analysis",
        func=financial_analysis,
        description="Perform financial analysis on a given stock."
    ),
    "Filings Analysis": Tool(
        name="Filings Analysis",
        func=filings_analysis,
        description="Analyze 10-Q and 10-K filings."
    ),
    "Investment Recommendation": Tool(
        name="Investment Recommendation",
        func=recommend,
        description="Provide investment recommendations."
    ),
    "Research": Tool(
        name="Research",
        func=lambda q: research(company_stock=q, user_prompt=research_prompt),
        description="Collect recent news and market sentiment."
    ),
    "Real-Time Price": Tool(
        name="Real-Time Price",
        func=get_real_time_stock_price,
        description="Get real-time stock price for a given ticker."
    ),
    "Sentiment Analysis": Tool(
        name="Sentiment Analysis",
        func=lambda q: TavilySearchResults(api_key=TAVILY_API_KEY).run(f"{q} stock market news sentiment analysis"),
        description="Analyze recent news sentiment for a stock."
    ),
}

# Initialize agents dynamically.
agents = {
    "Basic Info": initialize_agent([tools["Basic Stock Info"], tools["Tavily Search"]], llm, 
                                     AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Technical Analysis": initialize_agent([tools["Technical Analysis"], tools["Tavily Search"]], llm, 
                                             AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Financial Analysis": initialize_agent([tools["Financial Analysis"], tools["Tavily Search"]], llm, 
                                             AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Filings Analysis": initialize_agent([tools["Filings Analysis"], tools["Tavily Search"]], llm, 
                                           AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Research Analysis": initialize_agent([tools["Research"], tools["Tavily Search"]], llm, 
                                            AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "News Analysis": initialize_agent([tools["News Summary"], tools["Tavily Search"]], llm, 
                                        AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True),
    "Recommend": initialize_agent([tools["Investment Recommendation"], tools["Tavily Search"]], llm, 
                                    AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Real-Time Price": initialize_agent([tools["Real-Time Price"]], llm, 
                                          AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
    "Sentiment Analysis": initialize_agent([tools["Sentiment Analysis"]], llm, 
                                             AgentType.ZERO_SHOT_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors=True),
}

if "responses" not in st.session_state:
    st.session_state.responses = []

def get_analysis(analysis_type, query, callbacks=None):
    return agents[analysis_type].run(f"Perform {analysis_type.lower()} for {query}.", callbacks=callbacks)

# def ShowData():
#     stock_data = get_stock_price_chart(query, timeframe_mapping[timeframe])
#     current_price = get_latest_stock_price(query)
#     predicted_price = predict_next_stock_price(query, timeframe_mapping[timeframe])
    
#     if not stock_data.empty:
#         stock_data["Smoothed"] = stock_data["Close"].rolling(window=5, min_periods=1).mean()
        
#         # Graph 1: Historical Price Trend with Prediction using Matplotlib
#         fig1, ax1 = plt.subplots(figsize=(24, 16))
#         ax1.plot(stock_data.index, stock_data["Smoothed"], label="Smoothed Closing Price", 
#                  linewidth=3, color="DarkGreen", linestyle="-")
#         ax1.plot(stock_data.index, stock_data["Close"], label="Closing Price", 
#                  linewidth=2, color="Green", marker="o", markersize=6, alpha=0.8, 
#                  linestyle="-", markerfacecolor="white")
#         ax1.scatter(stock_data.index, stock_data["Close"], color="#81c784", s=50, alpha=0.7, zorder=3)
#         for idx, row in stock_data.iterrows():
#             ax1.annotate(f"${row['Close']:.2f}", (idx, row["Close"]), textcoords="offset points",
#                          xytext=(0, 5), ha="center", fontsize=8, color="black")
#         latest_date = stock_data.index[-1]
#         ax1.scatter(latest_date, current_price, color="red", s=120,
#                     label=f"Last Day Price: ${current_price}", edgecolors="black", zorder=4)
#         ax1.annotate(f"${current_price:.2f}", (latest_date, current_price), textcoords="offset points",
#                      xytext=(10, 10), ha="left", fontsize=12, color="red", fontweight="bold")
#         if predicted_price is not None:
#             if len(stock_data.index) > 1:
#                 delta = stock_data.index[-1] - stock_data.index[-2]
#                 next_date = stock_data.index[-1] + delta
#             else:
#                 next_date = stock_data.index[-1]
#             # Apply sentiment adjustment if selected.
#             if include_sentiment:
#                 adjusted_predicted, sentiment_scores = adjust_prediction_with_sentiment(predicted_price, query)
#                 display_price = adjusted_predicted
#                 sentiment_str = f"\nSentiment: {sentiment_scores['compound']:.2f}"
#             else:
#                 display_price = predicted_price
#                 sentiment_str = ""
#             ax1.axvline(x=next_date, color="purple", linestyle="--", label="Predicted Next Price")
#             ax1.annotate(
#                 f"Predicted: ${display_price:.2f}{sentiment_str}\non {next_date.strftime('%Y-%m-%d')}",
#                 (next_date, display_price),
#                 textcoords="offset points",
#                 xytext=(10, -10),
#                 ha="left", fontsize=12, color="purple", fontweight="bold"
#             )
#         ax1.set_facecolor("#f4f4f4")
#         ax1.set_title(f"{query} Stock Price Trend ({timeframe})", fontsize=18, fontweight="bold", color="#333")
#         ax1.set_xlabel("Date", fontsize=14)
#         ax1.set_ylabel("Price (USD)", fontsize=14)
#         ax1.tick_params(axis="x", rotation=45)
#         ax1.legend()
#         ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
#         ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
#         st.markdown(
#             """
#             <style>
#             .scrollable-container {
#                 overflow-x: auto;
#                 width: 100%;
#             }
#             </style>
#             """, unsafe_allow_html=True
#         )
#         st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
#         st.pyplot(fig1)
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Graph 2: Bar Chart Comparison for Overall Price
#         if predicted_price is not None:
#             fig2, ax2 = plt.subplots(figsize=(12, 12))
#             prices = [current_price, display_price]
#             labels = ["Last Day Price", "Predicted Next Price"]
#             colors = ["red", "purple"]
#             ax2.bar(labels, prices, color=colors, alpha=0.8)
#             for i, price in enumerate(prices):
#                 ax2.text(i, price + 0.5, f"${price:.2f}", ha="center", fontsize=12)
#             ax2.set_title(f"{query} Price Comparison", fontsize=16, fontweight="bold")
#             ax2.set_ylabel("Price (USD)")
#             st.pyplot(fig2)
#             st.metric(label=f"Last Day Price of {query}", value=f"${current_price}")
#             st.metric(label="Predicted Next Price", value=f"${display_price:.2f}")
#         else:
#             st.warning("Not enough data to predict the next stock price.")
        
#         # Graph 3: Bar Chart for Last 10 Days Actual vs Predicted
#         if len(stock_data) >= 30:
#             actual_prices = []
#             predicted_prices = []
#             dates = []
#             n = len(stock_data)
#             for i in range(n - 30, n):
#                 if i < 2:
#                     continue
#                 sub_data = stock_data.iloc[:i]
#                 X = np.arange(len(sub_data)).reshape(-1, 1)
#                 y = sub_data["Close"].values
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 pred = model.predict(np.array([[len(sub_data)]]))[0]
#                 predicted_prices.append(pred)
#                 actual_prices.append(stock_data["Close"].iloc[i])
#                 dates.append(stock_data.index[i])
#             if dates:
#                 fig3, ax3 = plt.subplots(figsize=(16, 12))
#                 x = np.arange(len(dates))
#                 width = 0.35
#                 bars1 = ax3.bar(x - width/2, actual_prices, width, label="Actual Price", color="red", alpha=0.8)
#                 bars2 = ax3.bar(x + width/2, predicted_prices, width, label="Predicted Price", color="purple", alpha=0.8)
#                 for bar in bars1:
#                     height = bar.get_height()
#                     ax3.annotate(f"${height:.2f}",
#                                  xy=(bar.get_x() + bar.get_width()/2, height),
#                                  xytext=(0, 3), textcoords="offset points",
#                                  ha="center", va="bottom", fontsize=10)
#                 for bar in bars2:
#                     height = bar.get_height()
#                     ax3.annotate(f"${height:.2f}",
#                                  xy=(bar.get_x() + bar.get_width()/2, height),
#                                  xytext=(0, 3), textcoords="offset points",
#                                  ha="center", va="bottom", fontsize=10)
#                 ax3.set_title(f"{query} Last 10 Days: Actual vs Predicted Prices", fontsize=16, fontweight="bold")
#                 ax3.set_ylabel("Price (USD)", fontsize=14)
#                 ax3.set_xticks(x)
#                 date_labels = [d.strftime("%b %d") for d in dates]
#                 ax3.set_xticklabels(date_labels, rotation=45)
#                 ax3.legend()
#                 st.pyplot(fig3)
                
#                 # Graph 4: Average of Last 10 Days Actual vs Predicted
#                 if actual_prices and predicted_prices:
#                     avg_actual = np.mean(actual_prices)
#                     avg_predicted = np.mean(predicted_prices)
#                     fig4, ax4 = plt.subplots(figsize=(12, 10))
#                     avg_labels = ["Avg Actual", "Avg Predicted"]
#                     avg_values = [avg_actual, avg_predicted]
#                     bars = ax4.bar(avg_labels, avg_values, color=["red", "purple"], alpha=0.8)
#                     for bar in bars:
#                         height = bar.get_height()
#                         ax4.annotate(f"${height:.2f}",
#                                      xy=(bar.get_x() + bar.get_width()/2, height),
#                                      xytext=(0, 3), textcoords="offset points",
#                                      ha="center", va="bottom", fontsize=12)
#                     ax4.set_title("Average of Last 10 Days: Actual vs Predicted", fontsize=16, fontweight="bold")
#                     ax4.set_ylabel("Price (USD)", fontsize=14)
#                     st.pyplot(fig4)
#                     st.metric(label="Avg Actual Price (Last 10 Days)", value=f"${avg_actual:.2f}")
#                     st.metric(label="Avg Predicted Price (Last 10 Days)", value=f"${avg_predicted:.2f}")
#             else:
#                 st.warning("Not enough data to generate last 10 days predictions comparison.")
#         else:
#             st.warning("Not enough data to generate last 10 days predictions comparison.")
        
#         # Graph 5: Predicted Next 30 Days Prices (using days_ahead=10 for next 10 days)
#         future_dates, future_predictions = predict_next_30_days(query, timeframe_mapping[timeframe], days_ahead=10)
#         st.write("Future Dates:", future_dates)
#         st.write("Future Predictions:", future_predictions)
#         if future_dates is not None and future_predictions is not None:
#             fig5, ax5 = plt.subplots(figsize=(24, 12))
#             ax5.plot(future_dates, future_predictions, marker="o", linestyle="-", color="blue", label="Predicted Price")
#             ax5.set_title(f"{query} Predicted Prices for Next 10 Days", fontsize=18, fontweight="bold")
#             ax5.set_xlabel("Date", fontsize=14)
#             ax5.set_ylabel("Predicted Price (USD)", fontsize=14)
#             ax5.tick_params(axis="x", rotation=45)
#             ax5.legend()
#             st.pyplot(fig5)
#         else:
#             st.warning("Not enough data to predict the next 10 days' prices.")
#     else:
#         st.warning(f"No data available for {query} for the selected timeframe.")





def ShowData():
    stock_data = get_stock_price_chart(query, timeframe_mapping[timeframe])
    current_price = get_latest_stock_price(query)
    predicted_price = predict_next_stock_price(query, timeframe_mapping[timeframe])
    
    if not stock_data.empty:
        # Calculate smoothed closing prices.
        stock_data["Smoothed"] = stock_data["Close"].rolling(window=5, min_periods=1).mean()
        
        ### Graph 1: Historical Price Trend with Prediction
        fig1 = go.Figure()
        # Smoothed closing price
        fig1.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data["Smoothed"],
            mode="lines",
            line=dict(color="darkgreen", width=3),
            name="Smoothed Closing Price"
        ))
        # Actual closing price (with markers)
        fig1.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data["Close"],
            mode="lines+markers",
            line=dict(color="green", width=2),
            marker=dict(size=6, symbol="circle-open"),
            name="Closing Price",
            text=[f"${val:.2f}" for val in stock_data["Close"]],
            textposition="top center",
            hovertemplate="%{text}"
        ))
        # Highlight the latest day price.
        latest_date = stock_data.index[-1]
        fig1.add_trace(go.Scatter(
            x=[latest_date],
            y=[current_price],
            mode="markers+text",
            marker=dict(color="red", size=12, line=dict(color="black", width=1)),
            text=[f"${current_price:.2f}"],
            textposition="top right",
            name=f"Last Day Price: ${current_price}"
        ))
        
        # If a predicted price is available, compute next_date and (optionally) apply sentiment adjustment.
        if predicted_price is not None:
            if len(stock_data.index) > 1:
                delta = stock_data.index[-1] - stock_data.index[-2]
                next_date = stock_data.index[-1] + delta
            else:
                next_date = stock_data.index[-1]
            if include_sentiment:
                adjusted_predicted, sentiment_scores = adjust_prediction_with_sentiment(predicted_price, query)
                display_price = adjusted_predicted
                sentiment_str = f"\nSentiment: {sentiment_scores['compound']:.2f}"
            else:
                display_price = predicted_price
                sentiment_str = ""
            # Add a vertical line for the predicted next price.
            fig1.add_shape(
                type="line",
                x0=next_date, x1=next_date,
                y0=min(stock_data["Close"].min(), display_price) * 0.95,
                y1=max(stock_data["Close"].max(), display_price) * 1.05,
                line=dict(color="purple", dash="dash")
            )
            # Annotate the predicted price.
            fig1.add_annotation(
                x=next_date,
                y=display_price,
                text=f"Predicted: ${display_price:.2f}{sentiment_str}<br>{next_date.strftime('%Y-%m-%d')}",
                showarrow=True,
                arrowhead=2,
                ax=10,
                ay=-40,
                font=dict(color="purple")
            )
        
        fig1.update_layout(
            title=f"{query} Stock Price Trend ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_white",
            xaxis=dict(tickformat="%b %d", tickangle=45)
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        ### Graph 2: Bar Chart Comparison for Overall Price
        if predicted_price is not None:
            fig2 = go.Figure(data=[
                go.Bar(name="Last Day Price", x=["Last Day Price"], y=[current_price], marker_color="red"),
                go.Bar(name="Predicted Next Price", x=["Predicted Next Price"], y=[display_price], marker_color="purple")
            ])
            fig2.update_layout(
                title=f"{query} Price Comparison",
                yaxis_title="Price (USD)",
                barmode="group",
                template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.metric(label=f"Last Day Price of {query}", value=f"${current_price}")
            st.metric(label="Predicted Next Price", value=f"${display_price:.2f}")
        else:
            st.warning("Not enough data to predict the next stock price.")
        
        ### Graph 3: Bar Chart for Last 30 Days Actual vs Predicted
        # (Using last 30 days data for a more robust comparison)
        if len(stock_data) >= 30:
            actual_prices = []
            predicted_prices = []
            date_labels = []
            n = len(stock_data)
            for i in range(n - 30, n):
                if i < 2:
                    continue
                sub_data = stock_data.iloc[:i]
                X = np.arange(len(sub_data)).reshape(-1, 1)
                y = sub_data["Close"].values
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(np.array([[len(sub_data)]]))[0]
                predicted_prices.append(pred)
                actual_prices.append(stock_data["Close"].iloc[i])
                date_labels.append(stock_data.index[i].strftime("%b %d"))
            if date_labels:
                fig3 = go.Figure(data=[
                    go.Bar(name="Actual Price", x=date_labels, y=actual_prices, marker_color="red"),
                    go.Bar(name="Predicted Price", x=date_labels, y=predicted_prices, marker_color="purple")
                ])
                fig3.update_layout(
                    title=f"{query} Last 30 Days: Actual vs Predicted Prices",
                    yaxis_title="Price (USD)",
                    barmode="group",
                    template="plotly_white"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                ### Graph 4: Average of Last 10 Days Actual vs Predicted
                if actual_prices and predicted_prices:
                    avg_actual = np.mean(actual_prices)
                    avg_predicted = np.mean(predicted_prices)
                    fig4 = go.Figure(data=[
                        go.Bar(name="Avg Actual", x=["Avg Actual"], y=[avg_actual], marker_color="red"),
                        go.Bar(name="Avg Predicted", x=["Avg Predicted"], y=[avg_predicted], marker_color="purple")
                    ])
                    fig4.update_layout(
                        title="Average of Last 30 Days: Actual vs Predicted",
                        yaxis_title="Price (USD)",
                        barmode="group",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                    st.metric(label="Avg Actual Price (Last 30 Days)", value=f"${avg_actual:.2f}")
                    st.metric(label="Avg Predicted Price (Last 30 Days)", value=f"${avg_predicted:.2f}")
            else:
                st.warning("Not enough data to generate last 30 days predictions comparison.")
        else:
            st.warning("Not enough data to generate last 30 days predictions comparison.")
        
        ### Graph 5: Predicted Next 30 Days Prices (using days_ahead=10 for next 10 days)
        future_dates, future_predictions = predict_next_5_days(query, timeframe_mapping[timeframe], days_ahead=5)
        #st.write("Future Dates:", future_dates)
        #st.write("Future Predictions:", future_predictions)
        if future_dates is not None and future_predictions is not None:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=future_dates, 
                y=future_predictions, 
                mode="lines+markers", 
                marker=dict(color="blue", size=8),
                line=dict(color="blue", width=2),
                name="Predicted Price"
            ))
            fig5.update_layout(
                title=f"{query} Predicted Prices for Next 5 Days",
                xaxis_title="Date",
                yaxis_title="Predicted Price (USD)",
                template="plotly_white",
                xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("Not enough data to predict the next 5 days' prices.")
    else:
        st.warning(f"No data available for {query} for the selected timeframe.")


if analyze_button:
    ShowData()
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
    st.info(f"Analyzing stock: {query}")
    
    try:
        if analysis_type == "Recommend":
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            basic_info = get_analysis("Basic Info", query)
            financial_info = get_analysis("Financial Analysis", query)
            sentiment_info = get_analysis("Sentiment Analysis", query)
            response = get_analysis(
                "Recommend",
                f"Using Basic Info:\n{basic_info}\n"
                f"Using Financial Analysis:\n{financial_info}\n"
                f"Using Sentiment Analysis:\n{sentiment_info}\n"
                f"Provide an investment recommendation for {query}.",
                callbacks=[st_cb]
            )
        else:
            response = get_analysis(analysis_type, query, callbacks=[st_cb])
            st.success(f"{analysis_type} Complete")
            st.session_state.responses.append({
                "query": query,
                "analysis_type": analysis_type,
                "response": response
            })
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
