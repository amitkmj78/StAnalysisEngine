import os
import datetime
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from sklearn.ensemble import GradientBoostingRegressor
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults

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
select_llm = st.sidebar.selectbox("Select LLM Type", ["Ollama-Local", "Open AI"], key="llm_select")
llm = llm_openai if select_llm == "Open AI" else llm_ollama

# Sidebar inputs
st.sidebar.header("Stock Query")
query = st.sidebar.text_input("Enter Stock ticker (e.g., AAPL):", value="AAPL", key="ticker_input")
prediction_days = st.sidebar.number_input("Prediction Days", min_value=1, max_value=365, value=5, step=1,
                                            help="Select the number of future trading days to predict.", key="pred_days")
# Earnings impact checkbox in sidebar.
include_earnings_impact = st.sidebar.checkbox("Include Earnings Impact using AI Agent(Reaserch)", value=True, key="earnings_checkbox")
analysis_type = st.sidebar.selectbox("Select AI Agent Type", 
    ["Research Analysis", "Basic Info", "Technical Analysis", "Financial Analysis", "Filings Analysis", 
     "News Analysis", "Recommend", "Real-Time Price", "Sentiment Analysis"], key="analysis_type")
timeframe = st.sidebar.radio("Select Timeframe:", list({
   
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Month": "6mo",
    "1 Year": "1y",
    "5 Year": "5y"
}.keys()), key="timeframe")
chart_type = st.sidebar.selectbox("Select Chart Type:", ["Line Chart", "Candlestick Chart"], key="chart_type")
include_sentiment = st.sidebar.checkbox("Include Sentiment Adjustment for Predicted Price", value=False, key="sentiment_checkbox")
analyze_button = st.sidebar.button("📊Analyze", key="analyze_button")
research_prompt = st.sidebar.text_input("Modify Prompt (Optional)", placeholder="Enter custom prompt...", key="research_prompt_input")

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

# Prediction function for next-day price.
def predict_next_stock_price(ticker, period):
    data = get_stock_price_chart(ticker, period)
    if data.empty or len(data) < 10:
        return None
    df = data.copy()
    df["Index"] = np.arange(len(df))
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Lag1"] = df["Close"].shift(1)
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()
    features = ["Index", "Lag1", "MA5", "MA10", "Return"]
    X = df[features].values
    y = df["Close"].values
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)
    next_index = df["Index"].iloc[-1] + 1
    last_close = df["Close"].iloc[-1]
    last_MA5 = df["MA5"].iloc[-1]
    last_MA10 = df["MA10"].iloc[-1]
    last_return = df["Return"].iloc[-1]
    X_new = np.array([[next_index, last_close, last_MA5, last_MA10, last_return]])
    predicted = model.predict(X_new)[0]
    return predicted

# Earnings Impact function using the research agent.
def get_earnings_impact(ticker):
    """
    Calls the research agent with a custom prompt to derive a sentiment score for the latest
    quarterly earnings of the given ticker. Expected output: a number between -1 (very negative)
    and 1 (very positive). Uses regex to extract the numeric value.
    """
    prompt = (
        f"Analyze the latest quarterly earnings report for {ticker}. "
        "Based on revenue growth, profit margins, and forward guidance, "
        "provide a sentiment score on a scale from -1 (very negative) to 1 (very positive). "
        "Return only the numeric score."
    )
    research_agent = initialize_agent(
        [tools["Research"], tools["Tavily Search"]],
        llm,
        AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    result = research_agent.run(prompt)
    #st.info("Research GEN AI Agent Output:  {result}")
    match = re.search(r"(-?\d+\.?\d*)", result)
    if match:
        try:
            impact = float(match.group(1))
        except Exception as e:
            st.error(f"Error parsing numeric value: {e}")
            impact = 0.0
    else:
        st.error("No numeric score found in the research agent output.")
        impact = 0.0
    return impact



# Function to predict next 30 days dynamically with recursive feature updating.
def predict_next_30_days(ticker, period, days_ahead=30):
    data = get_stock_price_chart(ticker, period)
    st.write("Historical Data:", data)
    if data.empty or len(data) < 20:
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

    features = ["Index", "Lag1", "MA3", "MA10", "Return", "Volatility", "Gap"]
    X = df[features].values
    y = df["Close"].values

    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)

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
        pred = model.predict(X_new)[0]
        predictions.append(pred)
        new_return = (pred - last_close) / last_close
        new_MA3 = (2/3) * last_MA3 + (1/3) * pred
        new_MA10 = (9/10) * last_MA10 + (1/10) * pred
        new_vol = 0.9 * last_vol + (0.1 * abs(new_return))
        new_gap = last_gap  # For simplicity, gap remains constant.
        last_close = pred
        last_return = new_return
        last_MA3 = new_MA3
        last_MA10 = new_MA10
        last_vol = new_vol
        last_gap = new_gap
   # Adjust predictions based on earnings impact if enabled.
    if include_earnings_impact:
        earnings_impact = get_earnings_impact(ticker)
        factor_earnings = 0.05  # Adjust as needed.
        st.info(f"Reaserch Agent AI Earnings Impact for {ticker}: {earnings_impact}")
        predictions = [p * (1 + factor_earnings * earnings_impact) for p in predictions]

    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=days_ahead)
    #st.write("Predicted Prices:", predictions)
    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions
    })
    prediction_df["Date"] = prediction_df["Date"].astype(str)
    st.dataframe(prediction_df)
    return future_dates, predictions

def get_sentiment_analysis(ticker):
    search_results = TavilySearchResults(api_key=TAVILY_API_KEY).run(f"{ticker} stock market news sentiment analysis")
    return f"Sentiment Analysis for {ticker}:\n\n{search_results}"

def adjust_prediction_with_sentiment(prediction, ticker):
    sentiment_text = get_sentiment_analysis(ticker)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentiment_text)
    compound = scores['compound']
    factor = 0.02
    adjusted = prediction * (1 + factor * compound)
    return adjusted, scores


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
# Updated timeframe mapping.
timeframe_mapping = {
    "1 Year": "1y",
    "1 Week": "5d",
    "30 Days": "1mo",
    "6 Month": "6mo",
    "5 Year": "5y"
}

if "responses" not in st.session_state:
    st.session_state.responses = []

def get_analysis(analysis_type, query, callbacks=None):
    return agents[analysis_type].run(f"Perform {analysis_type.lower()} for {query}.", callbacks=callbacks)

def ShowData():
    stock_data = get_stock_price_chart(query, timeframe_mapping[timeframe])
    current_price = get_latest_stock_price(query)
    predicted_price = predict_next_stock_price(query, timeframe_mapping[timeframe])
    
    if not stock_data.empty:
        stock_data["Smoothed"] = stock_data["Close"].rolling(window=5, min_periods=1).mean()
        
        ### Graph 1: Historical Price Trend with Prediction
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data["Smoothed"],
            mode="lines",
            line=dict(color="darkgreen", width=3),
            name="Smoothed Closing Price"
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
            hovertemplate="%{text}"
        ))
        latest_date = stock_data.index[-1]
        fig1.add_trace(go.Scatter(
            x=[latest_date],
            y=[current_price],
            mode="markers+text",
            marker=dict(color="#facff8", size=12, line=dict(color="black", width=1)),
            text=[f"${current_price:.2f}"],
            textposition="top right",
            name=f"Last Day Price: ${current_price}"
        ))
        
        if predicted_price is not None:
            if len(stock_data.index) > 1:
                delta = stock_data.index[-1] - stock_data.index[-2]
                next_date = stock_data.index[-1] + delta
            else:
                next_date = stock_data.index[-1]
            if include_sentiment:
                adjusted_predicted, sentiment_scores = adjust_prediction_with_sentiment(predicted_price, query)
                display_price = adjusted_predicted
                sentiment_str = f"\n AI Based Sentiment Score Based on MKT News: {sentiment_scores['compound']:.2f}"
                st.info(sentiment_str)
            else:
                display_price = predicted_price
                sentiment_str = ""
            fig1.add_shape(
                type="line",
                x0=next_date, x1=next_date,
                y0=min(stock_data["Close"].min(), display_price) * 0.95,
                y1=max(stock_data["Close"].max(), display_price) * 1.05,
                line=dict(color="purple", dash="dash")
            )
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
                go.Bar(name="Last Day Price", x=["Last Day Price"], y=[current_price], marker_color="#5DE2E7"),
                go.Bar(name="AI Predicted Next Price", x=["Predicted Next Price"], y=[display_price], marker_color="#4AFC5C")
            ])
            fig2.update_layout(
                title=f"{query} Price Comparison",
                yaxis_title="Price (USD)",
                barmode="group",
                template="plotly_white"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.metric(label=f"Last Day Price of {query}", value=f"${current_price}")
            st.metric(label="AI Predicted Next Price", value=f"${display_price:.2f}")
        else:
            st.warning("Not enough data to predict the next stock price.")
        
        ### Graph 3: Bar Chart for Last 30 Days Actual vs Predicted
        if len(stock_data) >= 30:
            actual_prices = []
            predicted_prices = []
            date_labels = []
            n = len(stock_data)
            for i in range(n - 30, n):
                if i < 3:
                    continue
                sub_data = stock_data.iloc[:i].copy()
                sub_data["Index"] = np.arange(len(sub_data))
                sub_data["Lag1"] = sub_data["Close"].shift(1)
                sub_data["MA3"] = sub_data["Close"].rolling(window=3).mean()
                sub_data["Return"] = sub_data["Close"].pct_change()
                sub_data = sub_data.dropna()
                if len(sub_data) < 2:
                    continue
                features = ["Index", "Lag1", "MA3", "Return"]
                X = sub_data[features].values
                y = sub_data["Close"].values
                temp_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=4, random_state=42)
                temp_model.fit(X, y)
                last_index_sub = sub_data["Index"].iloc[-1] + 1
                last_lag1 = sub_data["Close"].iloc[-1]
                last_ma3 = sub_data["MA3"].iloc[-1]
                last_return = sub_data["Return"].iloc[-1]
                X_new = np.array([[last_index_sub, last_lag1, last_ma3, last_return]])
                pred = temp_model.predict(X_new)[0]
                predicted_prices.append(pred)
                actual_prices.append(stock_data["Close"].iloc[i])
                date_labels.append(stock_data.index[i].strftime("%b %d"))
            
            if date_labels:
                fig3 = go.Figure(data=[
                    go.Bar(name="Actual Price", x=date_labels, y=actual_prices, marker_color="#5DE2E7"),
                    go.Bar(name="Predicted Price", x=date_labels, y=predicted_prices, marker_color="#4AFC5C")
                ])
                fig3.update_layout(
                    title=f"{query} Last 30 Days: Actual vs Predicted Prices",
                    yaxis_title="Price (USD)",
                    barmode="group",
                    template="plotly_white"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                ### Graph 4: Average of Last 30 Days Actual vs Predicted
                if actual_prices and predicted_prices:
                    avg_actual = np.mean(actual_prices)
                    avg_predicted = np.mean(predicted_prices)
                    fig4 = go.Figure(data=[
                        go.Bar(name="Avg Actual", x=["Avg Actual"], y=[avg_actual], marker_color="#5DE2E7"),
                        go.Bar(name="Avg Predicted", x=["Avg Predicted"], y=[avg_predicted], marker_color="#4AFC5C")
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
                
                ### Graph 5: Predicted Next 30 Days Prices (Dynamic)
                future_dates, future_predictions = predict_next_30_days(query, timeframe_mapping[timeframe], days_ahead=prediction_days)
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
                        title=f"{query} GEN AI Agent Based Predicted Prices for Next {prediction_days} Days",
                        xaxis_title="Date",
                        yaxis_title="Predicted Price (USD)",
                        template="plotly_white",
                        xaxis=dict(tickangle=45)
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                else:
                    st.warning("Not enough data to predict the next 30 days' prices.")
            else:
                st.warning("Not enough data to generate last 30 days predictions comparison.")
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
            st.success(f"{analysis_type} Complete")
            st.session_state.responses.append({
                "query": query,
                "analysis_type": analysis_type,
                "response": response
            })
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
