import streamlit as st
from services.config import validate_environment
from services.llm_setup import init_llms
from Agent.meta_agent import build_agent

st.set_page_config(page_title="AI Stock Intelligence Dashboard", layout="wide")

# -------------------------------------------------------------
#  ENV VALIDATION
# -------------------------------------------------------------
validate_environment()

# -------------------------------------------------------------
#  SIDEBAR — GLOBAL SETTINGS
# -------------------------------------------------------------
st.sidebar.title("⚙️ Global Settings")

# Theme
theme = st.sidebar.radio("Theme", ["Dark Mode", "Light Mode"])
st.session_state["theme"] = theme
PLOTLY_THEME = "plotly_dark" if theme == "Dark Mode" else "plotly_white"

# Risk Settings
risk_profile = st.sidebar.selectbox("Risk Profile", ["Aggressive", "Balanced", "Conservative"])
risk_factor = st.sidebar.slider("Risk Factor", 1, 10, 5)

st.session_state["risk_profile"] = risk_profile
st.session_state["risk_factor"] = risk_factor

# Ticker
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
st.session_state["ticker"] = ticker

# Timeframe
from services.data_service import TIMEFRAME_MAPPING
timeframe = st.sidebar.radio("Historical Timeframe", list(TIMEFRAME_MAPPING.keys()))
st.session_state["timeframe"] = timeframe

# -------------------------------------------------------------
# LLM Setup
# -------------------------------------------------------------
llm_openai, llm_groq, llm_labels = init_llms()

if not llm_labels:
    st.error("❌ No LLM models available.")
    st.stop()

model_choice = st.sidebar.selectbox("Choose LLM", llm_labels)
llm = llm_groq if model_choice.startswith("Groq") else llm_openai

if "meta_agent" not in st.session_state or st.session_state["meta_agent_model"] != model_choice:
    st.session_state["meta_agent"] = build_agent(llm)
    st.session_state["meta_agent_model"] = model_choice
    st.toast(f"Meta-Agent rebuilt using {model_choice}")

# -------------------------------------------------------------
# HOME PAGE CONTENT
# -------------------------------------------------------------
st.title("📊 AI Stock Intelligence Dashboard")

st.markdown("""
Welcome!  
Please use the left sidebar to choose ticker, timeframe, LLM model, and risk settings.

Navigate using the **Pages** menu on the left.
""")
