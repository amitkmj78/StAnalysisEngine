import streamlit as st
from services.config import validate_environment
from services.llm_setup import init_llms
from services.ui_service import apply_app_shell
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
llm_openai, llm_groq, llm_claude, llm_ollama, llm_labels = init_llms()

if not llm_labels:
    st.error("❌ No LLM models available.")
    st.stop()

model_choice = st.sidebar.selectbox("Choose LLM", llm_labels)
if model_choice.startswith("Groq"):
    llm = llm_groq
elif model_choice.startswith("Claude"):
    llm = llm_claude
elif model_choice.startswith("Local"):
    llm = llm_ollama
else:
    llm = llm_openai

if "meta_agent" not in st.session_state or st.session_state["meta_agent_model"] != model_choice:
    st.session_state["meta_agent"] = build_agent(llm)
    st.session_state["meta_agent_model"] = model_choice
    st.toast(f"Meta-Agent rebuilt using {model_choice}")

# -------------------------------------------------------------
# HOME PAGE CONTENT
# -------------------------------------------------------------
apply_app_shell(
    "AI Stock Intelligence Dashboard",
    "A cleaner app-style workspace for finding strong index funds and surfacing stock ideas for short-term or long-term investing.",
)

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Index fund ranking</span>
    <span class="app-pill">ETF comparison</span>
    <span class="app-pill">Goal-based scoring</span>
    <span class="app-pill">Shortlist builder</span>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1.35, 1])

with col1:
    st.markdown(
        """
<div class="app-card">
    <h3>Start Here</h3>
    <p>Use the left sidebar for your basic settings, then open the <strong>Best Index Fund</strong> page to compare major ETFs in one focused flow.</p>
    <div class="app-pill-row">
        <span class="app-pill">Index Funds</span>
        <span class="app-pill">VOO</span>
        <span class="app-pill">VTI</span>
        <span class="app-pill">QQQ</span>
        <span class="app-pill">BND</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-card">
    <h3>Best Index Fund Finder</h3>
    <p>Compare major ETFs like <strong>VOO</strong>, <strong>VTI</strong>, <strong>IVV</strong>, <strong>QQQ</strong>, <strong>VXUS</strong>, and <strong>BND</strong> in one place.</p>
    <p>Rank them by balanced core holding, lowest cost, best growth, or most stable option.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-card">
    <h3>Best Stock Finder</h3>
    <p>Screen stocks for <strong>short-term</strong> momentum or <strong>long-term</strong> investing using the same shortlist-first app flow.</p>
    <p>Compare returns, volatility, drawdown, growth, valuation, RSI, MACD, and volume strength.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-card">
    <h3>Monthly Investing Plan</h3>
    <p>Model what investing <strong>$1,000 every month</strong> into a top-ranked fund or stock could look like over time.</p>
    <p>Use it to compare a disciplined monthly plan with both historical results and a simple forward estimate.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-card">
    <h3>Strategies</h3>
    <p>Build a target-based investing strategy with dynamic years, target amount, and return assumptions.</p>
    <p>See the best fund and best stock candidates that can help support the plan.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="app-card">
    <h3>Best To Enter Now</h3>
    <p>Scan funds or stocks for the strongest current entry setups, or switch to one-ticker mode for a deeper buy-zone view.</p>
    <p>Useful when you want to know what looks actionable right now and exactly how to approach the entry.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
<div class="app-card">
    <h3>What It Does</h3>
    <p>Ranks major index ETFs and stocks using focused scoring models built for different investing goals and time horizons.</p>
    <p>All data on the new finder and monthly plan pages is pulled from <strong>Yahoo Finance via yfinance</strong>.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="app-card">
    <h3>Quick Tip</h3>
    <p>Funds tracking the same index can be nearly interchangeable. In that case, expense ratio, liquidity, and account availability usually decide the final pick.</p>
</div>
""",
        unsafe_allow_html=True,
    )
