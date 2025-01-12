
import os
import datetime
from langchain.chat_models import ChatOpenAI
from Agent.newAgent import news_summary
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from Agent.basicAgent import get_basic_stock_info
from Agent.technicalAgent import get_technical_analysis
from Agent.filingAgent import filings_analysis
from Agent.financialAgent import financial_analysis
from Agent.recommendAgent import recommend
from Agent.reasearchAgent import research
from langchain.agents import AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
import io
from dotenv import load_dotenv
from langchain.tools import Tool


load_dotenv()

# Set up conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

today_date = datetime.date.today()
st.info(f"Today Date : {today_date}")

search = TavilySearchResults(api_key="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",max_results=2)

 # API Key setup (replace with actual API key or use environment variables)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq model
llmollama = ChatGroq(model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

# API Key setup (replace with actual API key or use environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

# Initialize OpenAI LLM
#llmopenai = ChatOpenAI( openai_api_key=OPENAI_API_KEY)
llmopenai = ChatOpenAI(model="gpt-4o", temperature=0.7)  

news_tool = Tool(
    name="News Summary Tool",
    func=news_summary,
    description="Fetches and summarizes the latest news for a given query."
)

# Initialize TavilySearchResults
def tavily_search(query: str) -> str:
    """
    Perform a Tavily search for the given query.
    """
    search = TavilySearchResults(api_key=TAVILY_API_KEY)
    return search.run(query)

tavily_search_tool = Tool(
    name="Tavily Search Tool",
    func=tavily_search,
    description="Fetches real-time web search results using Tavily."
)

# Define tools for LangChain
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


# Define tools
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
# Combine tools into agents
#tools = [financial_analysis_tool, research_tool, filings_analysis_tool, recommend_tool,basic_stock_info_tool,technical_analysis_tool]


selectllm = st.sidebar.selectbox(
    "Select LLM Type",
    ["Ollama- Local", "Open AI"]
)
if selectllm=="Open AI":
    llm=llmopenai
else:
    llm=llmollama



# Initialize agents with tools
financial_analysis_agent= initialize_agent(
    tools=[financial_analysis_tool, technical_analysis_tool, news_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    max_iterations=3  # Limit to 3 steps to avoid infinite loops
)

# Initialize agents with tools
research_agent = initialize_agent(
    tools=[research_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
# Initialize agents with tools
filings_analysis_agent = initialize_agent(
    tools=[filings_analysis_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
# Initialize agents with tools
recommend_agent = initialize_agent(
    tools=[recommend_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
# Initialize agents with tools
basic_stock_info_agent = initialize_agent(
    tools=[basic_stock_info_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)
# Initialize agents with tools
technical_analysis_agent = initialize_agent(
    tools=[technical_analysis_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    StopIteration=True,
    memory=memory    
)

# Initialize agents with tools
New_analysis_agent = initialize_agent(
    tools=[news_tool,tavily_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    StopIteration=True,
    memory=memory    
)

# Initialize session state for response history
if "responses" not in st.session_state:
    st.session_state.responses = []

# Streamlit app
#st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard")

st.sidebar.header("Stock Query")
query = st.sidebar.text_area("Enter stock ticker (e.g., AAPL):", "AAPL")


analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Basic Info", "Technical Analysis","Financial Analysis", "Reaserach Analysis","Filings Analysis","Recommend", "News Analysis"]
)
analyze_button = st.sidebar.button("Analyze")

if analyze_button:
    st.info(f"Analyzing stock: {query}")
    try:
        if analysis_type == "Basic Info":
            response = basic_stock_info_agent.run(f"Provide the basic Detail Analysis {query}.")
            st.success("Basic Analysis Complete")
          
        elif analysis_type == "Technical Analysis":
             response = technical_analysis_agent.run(f"Provide the Technical Analysis {query}.")
             st.success( " Technical Analysis Complete")
             #st.markdown(response)
        elif analysis_type=="Recommend":
             response = recommend_agent.run(f"Provide an investment recommendation based on all Analysis {query}.")
             st.success( " Recomand Analysis Complete")
             #st.markdown(response)
        elif analysis_type=="Filings Analysis":
             response = filings_analysis_agent.run(f"Analyze the latest filings for {query}.")
             verbose_output = io.StringIO()
             verbose_log = verbose_output.getvalue()
             st.success( "Filings Analysis Complete")
             #st.markdown(response)
        elif analysis_type=="Reaserach Analysis":
             response = research_agent.run(f"Collect recent news and sentiments for {query}.")
             st.success( "Reaserach Analysis Complete")
             #st.markdown(response)
        elif analysis_type=="Financial Analysis":
             response = filings_analysis_agent.run(f"Perform financial analysis for {query}.")
             st.success( "Financial  Complete")
             #st.markdown(response)
        elif analysis_type=="News Analysis":
             response = New_analysis_agent.run(f"Perform News  analysis for {query}.")
             st.success( "New Analysis  Complete")
             #st.markdown(response)

             #New_analysis_agnet
        st.session_state.responses.append({"query": query, "analysis_type": analysis_type, "response": response})

        st.success("Analysis Complete!")
             
    except Exception as e:
        st.error(f"An error occurred: {e}")



st.markdown("### Analysis History")
if st.session_state.responses:
    for entry in st.session_state.responses:
        st.markdown(f"**Query:** {entry['query']}")
        st.markdown(f"**Analysis Type:** {entry['analysis_type']}")
        st.markdown(f"**Response:**\n{entry['response']}")
       # st.markdown(f"**Agent Activity Log:**")
        #st.code(entry["verbose_log"], language="plaintext")
        st.markdown("---")
else:
    st.write("No analysis has been run yet.")

