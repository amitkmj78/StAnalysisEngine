# Import Libraries
import os
import pandas as pd
import yfinance as yf
import datetime
#from langchain.prompts import PromptTemplate
from langchain_community.llms.openai import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
#import io
import warnings


warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


# Suppress warnings


# Initialize Memory
chat_history = ChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)


# Set up Streamlit page
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")
today_date = datetime.date.today()
st.info(f"Today Date : {today_date}")
# Initialize LLMs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llmollama = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

#llmollama = ChatGroq(model_name="llama3.3", groq_api_key=GROQ_API_KEY)

llmopenai = OpenAI(openai_api_key=OPENAI_API_KEY)

# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Select LLM
selectllm = st.sidebar.selectbox(
    "Select LLM Type",
    ["Ollama- Local", "Open AI"]
)
llm = llmopenai if selectllm == "Open AI" else llmollama


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def get_basic_stock_info(ticker: str) -> str:
    """
    Fetches and returns a detailed overview of a company's stock information for a given ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    str: A formatted string containing the company's basic stock information or an error message.
    """
    try:
        # Fetch stock information
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"No data found for the ticker {ticker}."

        # Construct the detailed stock overview
        return f"""
        ** Stock Information for {ticker} as of {today_date}: **
        
        - ** Name: ** {info.get('longName', 'N/A')}
        - ** Sector: ** {info.get('sector', 'N/A')}
        - ** Industry: ** {info.get('industry', 'N/A')}
        - ** Current Price: ** ${info.get('currentPrice', 'N/A')}
        - ** Market Cap: ** {info.get('marketCap', 'N/A')}
        - ** Full-Time Employees: ** {info.get('fullTimeEmployees', 'N/A')}
        - ** Enterprise Value: ** {info.get('enterpriseValue', 'N/A')}
        - ** 200-Day Average: ** ${info.get('twoHundredDayAverage', 'N/A')}
        - ** 52-Week High/Low: ** ${info.get('fiftyTwoWeekHigh', 'N/A')} / ${info.get('fiftyTwoWeekLow', 'N/A')}
        - ** Trailing P/E: ** {info.get('trailingPE', 'N/A')}
        - ** Forward P/E: ** {info.get('forwardPE', 'N/A')}
        - ** EBITDA: ** {info.get('ebitda', 'N/A')}
        - ** Total Revenue: ** {info.get('totalRevenue', 'N/A')}
        - ** Revenue Per Share: ** ${info.get('revenuePerShare', 'N/A')}
        - ** Operating Cashflow: ** {info.get('operatingCashflow', 'N/A')}

        ** ---Highlights:   **
        - nRecent trading activity includes notable price fluctuations:
          - ** Previous Close: ** ${info.get('previousClose', 'N/A')}
          - ** Day Range (Low/High): ** ${info.get('regularMarketDayLow', 'N/A')} / ${info.get('regularMarketDayHigh', 'N/A')}
        -  Analyst targets:
          - ** Target High Price: ** ${info.get('targetHighPrice', 'N/A')}
          - ** Target Low Price: ** ${info.get('targetLowPrice', 'N/A')}
          - ** Target Mean Price: ** ${info.get('targetMeanPrice', 'N/A')}

        This information reflects the most recent data available for {ticker}.
        """
    except Exception as e:
        # Handle exceptions and return a formatted error message
        return f"An error occurred while fetching data for {ticker}: {str(e)}"

def get_technical_analysis(ticker: str, period: str = "1y") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    st.write(ticker)
    #st.write(stock)
    history = stock.history(period=period)
    #st.write(history)
    if history.empty:
        return pd.DataFrame({"Error": ["No historical data available"]})
    
    history['SMA_50'] = history['Close'].rolling(window=50).mean()
    history['RSI'] = calculate_rsi(history['Close'])
    latest = history.iloc[-1]
    return pd.DataFrame({
        'Indicator': ['Current Price', '50-day SMA', 'RSI (14-day)'],
        'Value': [
            f"${latest['Close']: .2f} ",
            f"${latest['SMA_50']: .2f} ",
            f"{latest['RSI']: .2f} "
        ]
    })

def financial_analysis (company_stock: str):
 return f"""
          Conduct a thorough analysis of {company_stock}'s by today date { today_date} stock financial health and market performance. 
        This includes examining key financial metrics such as P/E ratio, EPS growth, revenue trends, and 
        debt-to-equity ratio. Also, analyze the stock's performance in comparison to its industry peers and 
        overall market trends.
    
        expected_output:
        The final report must expand on the summary provided but now including a clear assessment of the stock's 
        financial standing, its strengths and weaknesses, and how it fares against its competitors in the current 
        market scenario. Make sure to use the most recent data possible.
    """


def research(company_stock: str):
    return f"""
    Collect and summarize recent news as of today date {today_date} articles, press releases, and market analyses 
    related to the {{company_stock}} stock and its industry. Pay special attention to any significant events, 
    market sentiments, and analysts' opinions. Also include upcoming events like earnings and others. 
    expected_output: 
     A report that includes a comprehensive summary of the latest news, including today's date ({today_date}), 
    any notable shifts in market sentiment, and potential impacts on the stock. Also, make sure to return the stock 
    ticker as {company_stock}. Make sure to use the most recent data possible.   

    """

def filings_analysis(company_stock: str):
    return f"""
        Analyze the latest 10-Q and 10-K filings from EDGAR for the stock {company_stock} as of today {today_date}. 
        Focus on key sections like Management's Discussion and Analysis, financial statements, insider trading activity, 
        and any disclosed risks. Extract relevant data and insights that could influence the stock's future performance.
    
    expected_output:
        Final answer must be an expanded report that now also highlights significant findings 
        from these filings, including any red flags or positive indicators for your customer.
    """


def recommend(company_stock: str):
    
    return f"""
    **Investment Recommendation Report**  
    **Stock:** {company_stock}  
    **Date:** {today_date}  

    ---  

    **Executive Summary:**  
    This report synthesizes analyses provided by the Financial Analyst and the Research Analyst. It integrates insights into the financial health, market sentiment, and qualitative data extracted from EDGAR filings.  

    ---  

    **Section 1: Financial Analysis**  
    - **Key Metrics:** (Include PE ratio, EPS, revenue growth, profit margins, etc.)  
    - **Balance Sheet Health:** (Assess liquidity, debt levels, and asset quality.)  
    - **Cash Flow Trends:** (Highlight operating, investing, and financing cash flows.)  

    **Section 2: Market Sentiment**  
    - **Recent Stock Performance:** (Evaluate price movements, volatility, and trends.)  
    - **Analyst Ratings:** (Summarize ratings, price targets, and consensus opinions.)  
    - **News Sentiment:** (Include notable headlines and sentiment analysis.)  

    **Section 3: Qualitative Insights from EDGAR Filings**  
    - **Key Disclosures:** (Discuss management commentary, risks, and strategies.)  
    - **Material Changes:** (Identify significant changes in operations or outlook.)  
    - **Other Highlights:** (Extract any unique insights or observations.)  

    **Section 4: Insider Trading Activity**  
    - **Recent Transactions:** (Summarize insider buying/selling activity.)  
    - **Implications:** (Discuss whether this reflects confidence or caution.)  

    **Section 5: Upcoming Events**  
    - **Earnings Report Date:** (Include relevant dates and anticipated impacts.)  
    - **Dividends/Buybacks:** (Note any announced plans.)  
    - **Corporate Events:** (Highlight conferences, product launches, etc.)  

    ---  

    **Recommendation:**  
    Based on the analysis above, our recommendation for {company_stock} is:  
    - **Investment Stance:** (e.g., Buy, Hold, or Sell)  
    - **Rationale:** (Summarize key factors supporting this stance.)  
    - **Strategy:** (Provide actionable advice, such as entry/exit points, long-term expectations, and risk management tips.)  

    ---  

    **Supporting Evidence:**  
    - Include relevant charts, graphs, or tables to enhance understanding.  
    - Ensure the report is professional and visually appealing for the customer.  

    ---  

    **Note:**  
    This recommendation is based on the data available as of {today_date}. Market conditions may change, and it is advised to reassess periodically.  

    """


# Define tools
financial_analysis_tool = Tool(
    name="Financial Analysis",
    func=financial_analysis,
    description="Perform financial analysis on a given stock, including financial metrics, comparisons, and insights."
)

research_tool = Tool(
    name="Research",
    func=research,
    description="Collect and summarize recent news, press releases, and market sentiment for a given stock."
)

filings_analysis_tool = Tool(
    name="Filings Analysis",
    func=filings_analysis,
    description="Analyze 10-Q and 10-K filings for key insights, risks, and indicators."
)

recommend_tool = Tool(
    name="Investment Recommendation",
    func=recommend,
    description="Review analyses and provide a comprehensive investment recommendation."
)

basic_stock_info_tool = Tool(
    name="Basic Stock Info",
    func=get_basic_stock_info,
    description="Fetch basic stock information for a given ticker symbol."
)

technical_analysis_tool = Tool(
    name="Technical Analysis",
    func=get_technical_analysis,
    description="Perform technical analysis for a given ticker symbol."
)

# Initialize agents
tools = {
    "Basic Info": [basic_stock_info_tool],
    "Technical Analysis": [technical_analysis_tool],
    "Financial Analysis": [financial_analysis_tool],
    "Research Analysis": [research_tool],
    "Filings Analysis": [filings_analysis_tool],
    "Recommend": [recommend_tool]
}

agents = {name: initialize_agent(
    tools=tools_list,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
     
) for name, tools_list in tools.items()}

# Session state for responses
if "responses" not in st.session_state:
    st.session_state.responses = []

# Sidebar input
st.sidebar.header("Stock Query")
query = st.sidebar.text_area("Enter stock ticker (e.g., AAPL):", "AAPL")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    list(tools.keys())
)
analyze_button = st.sidebar.button("Analyze")

# Perform analysis
if analyze_button:
    st.info(f"Analyzing stock: {query}")
    try:
        agent = agents[analysis_type]
        #verbose_output = io.StringIO()
        response = agent.run(f"Analyze {query} with focus on {analysis_type}.")
        #verbose_log = verbose_output.getvalue()
        st.session_state.responses.append({"query": query, "analysis_type": analysis_type, "response": response})
        st.success(f"{analysis_type} Analysis Complete")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display analysis history
st.markdown("### Analysis History")
if st.session_state.responses:
    for entry in st.session_state.responses:
        st.markdown(f"**Query:** {entry['query']}")
        st.markdown(f"**Analysis Type:** {entry['analysis_type']}")
        st.markdown(f"**Response:**\n{entry['response']}")
        #st.markdown("**Agent Activity Log:**")
        #st.code(verbose_log, language="plaintext") 
        st.markdown("---")
else:
    st.write("No analysis has been run yet.")
