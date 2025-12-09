# app.py
import time
import streamlit as st
import plotly.graph_objects as go
from services.config import validate_environment
from services.llm_setup import init_llms
from services.data_service import TIMEFRAME_MAPPING, get_stock_data, get_latest_price
from services.prediction_service import predict_next_price
from services.analysis_service import get_analysis
from services.websocket_service import start_price_feed, is_ws_configured
from Agent.meta_agent import build_agent, ask_meta_agent
from langchain_core.tools import Tool

# Animated Theme Toggle CSS
toggle_css = """
<style>
.toggle-container {
    display: flex;
    align-items: center;
    gap: 10px;
}
.switch {
  position: relative;
  display: inline-block;
  width: 48px;
  height: 24px;
}
.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}
.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #bbb;
  transition: .4s;
  border-radius: 24px;
}
.slider:before {
  position: absolute;
  content: "🌞";
  height: 20px;
  width: 20px;
  left: 2px;
  bottom: 2px;
  transition: .4s;
  font-size: 14px;
}
input:checked + .slider {
  background-color: #4AFC5C;
}
input:checked + .slider:before {
  transform: translateX(24px);
  content: "🌙";
}
</style>
"""
st.markdown(toggle_css, unsafe_allow_html=True)

# ============================
# 🎨 Theme Toggle (Fixed)
# ============================

# Save theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "Dark Mode"

theme = st.sidebar.radio(
    "🎨 UI Theme",
    ("Dark Mode", "Light Mode"),
    index=0 if st.session_state.theme == "Dark Mode" else 1,
)

st.session_state.theme = theme

# Apply theme style
if theme == "Dark Mode":
    app_bg = "#0E1117"
    sidebar_bg = "#111827"
    text_color = "#FFFFFF"
    metric_bg = "#111827"
    border_color = "#222222"
else:
    app_bg = "#FFFFFF"
    sidebar_bg = "#F7F7F7"
    text_color = "#000000"
    metric_bg = "#FFFFFF"
    border_color = "#DDDDDD"

# Inject CSS to all UI elements
dark_light_css = f"""
<style>
/* Main App Background */
.stApp {{
    background-color: {app_bg} !important;
    color: {text_color} !important;
}}

/* Sidebar Panel */
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    color: {text_color} !important;
}}

/* Metric Box Styling */
[data-testid="stMetric"] {{
    background-color: {metric_bg} !important;
    border-radius: 12px !important;
    padding: 14px !important;
    border: 1px solid {border_color} !important;
    color: {text_color} !important;
}}

/* Tabs + Headers */
h1, h2, h3, h4, h5, h6, p, label, span {{
    color: {text_color} !important;
}}
</style>
"""
st.markdown(dark_light_css, unsafe_allow_html=True)


# Optional: voice capture (if installed)
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

# -------------------------------------------------------------
# Page Config + Dark TradingView-style theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Intelligence",
    layout="wide",
    page_icon="📈",
)

DARK_CSS = """
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
div[data-testid="stMetric"] {
    background-color: #111827;
    border-radius: 10px;
    padding: 10px;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# Environment validation
# -------------------------------------------------------------
validate_environment()

# -------------------------------------------------------------
# LLM Setup & Meta-Agent
# -------------------------------------------------------------
llm_openai, llm_groq, llm_labels = init_llms()

if not llm_labels:
    st.error("❌ No usable LLM instance could be created. Double-check your API keys.")
    st.stop()

select_llm = st.sidebar.selectbox("Select LLM Model", llm_labels, key="llm_select")
if select_llm == "Groq Llama3-70B":
    llm = llm_groq
else:
    llm = llm_openai

# Build / cache meta-agent
if "meta_agent" not in st.session_state or st.session_state.get("meta_agent_model") != select_llm:
    st.session_state.meta_agent = build_agent(llm)
    st.session_state.meta_agent_model = select_llm

meta_agent = st.session_state.meta_agent

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.header("Stock Controls")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL").strip().upper()

timeframe = st.sidebar.radio(
    "Historical Timeframe:",
    list(TIMEFRAME_MAPPING.keys()),
    index=2,
    key="timeframe",
)

prediction_days = st.sidebar.number_input(
    "Prediction Days (for label only here)",
    min_value=1,
    max_value=60,
    value=5,
    step=1,
    key="pred_days",
)

enable_live_feed = st.sidebar.checkbox(
    "Enable Live WebSocket Price Feed",
    value=False,
    key="enable_live_feed",
)

analysis_type = st.sidebar.selectbox(
    "Single-Tool Analysis Type",
    [
        "Basic Info",
        "Technical Analysis",
        "Financial Analysis",
        "Filings Analysis",
        "News Analysis",
        "Recommend",
        "Real-Time Price",
        "Sentiment Analysis",
        "Research Analysis",
    ],
    key="analysis_type",
)

research_prompt = st.sidebar.text_input(
    "Custom Research Prompt (Optional)",
    placeholder="Custom prompt for Research Analysis…",
    key="research_prompt_input",
)

run_single_analysis = st.sidebar.button("Run Single Analysis", key="run_single_analysis")

# -------------------------------------------------------------
# Session State for Chat
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

if "responses" not in st.session_state:
    st.session_state.responses = []

# -------------------------------------------------------------
# Tabs Layout
# -------------------------------------------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📈 Charts & Prediction",
    "📊 Technical Indicators",
    "⏱ Live Price Feed",
    "🧠 Analyst AI",
    "💬 Chat & History",
])

# -------------------------------------------------------------
# Helper: chart rendering
# -------------------------------------------------------------
def render_price_chart(ticker: str, timeframe_label: str):
    period = TIMEFRAME_MAPPING[timeframe_label]
    data = get_stock_data(ticker, period)
    if data.empty:
        st.warning(f"No data available for {ticker} for timeframe {timeframe_label}.")
        return None, None

    latest_price = get_latest_price(ticker)
    predicted_price = predict_next_price(ticker, period)

    fig = go.Figure()

    # line chart / candlestick hybrid (simple)
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
        )
    )

    if latest_price is not None:
        fig.add_hline(
            y=latest_price,
            line_dash="dash",
            annotation_text=f"Last Close: ${latest_price:.2f}",
            annotation_position="top left",
        )

    if predicted_price is not None:
        fig.add_hline(
            y=predicted_price,
            line_dash="dot",
            line_color="purple",
            annotation_text=f"Predicted Next: ${predicted_price:.2f}",
            annotation_position="bottom left",
        )

    fig.update_layout(
        title=f"{ticker} Price Chart ({timeframe_label})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    return latest_price, predicted_price

# -------------------------------------------------------------
# 🏠 Overview Tab
# -------------------------------------------------------------
with tabs[0]:
    st.title("📊 AI Stock Intelligence Dashboard")
    if ticker:
        st.subheader(f"Overview — {ticker}")
        last_price = get_latest_price(ticker)
        col1, col2 = st.columns(2)
        with col1:
            if last_price is not None:
                st.metric("Last Close Price", f"${last_price:.2f}")
            else:
                st.metric("Last Close Price", "N/A")
        with col2:
            period = TIMEFRAME_MAPPING[timeframe]
            predicted_price = predict_next_price(ticker, period)
            if predicted_price is not None:
                st.metric("Predicted Next Price", f"${predicted_price:.2f}")
            else:
                st.metric("Predicted Next Price", "N/A")

        st.markdown("### Quick Company Snapshot")
        try:
            snapshot = get_analysis("Basic Info", ticker)
            st.write(snapshot)
        except Exception:
            st.warning("Unable to load basic info right now.")
    else:
        st.info("Enter a ticker in the sidebar to get started.")
    

# -------------------------------------------------------------
# 📈 Charts & Prediction Tab
# -------------------------------------------------------------
from services.prediction_service import (
    predict_backtest_prices,
    compute_backtest_metrics,
    predict_future_prices,
    generate_trading_signal,
)
from services.data_service import get_latest_price
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np

# Use current theme for all charts in this tab
plotly_theme = (
    "plotly_dark"
    if st.session_state.get("theme", "Dark Mode") == "Dark Mode"
    else "plotly_white"
)

# ---------- Simple Trading Simulation (LONG-ONLY) ----------
def run_trading_sim(backtest_df: pd.DataFrame, threshold_pct: float = 0.0):
    """
    Very simple strategy:
    - Go long at close(t) if model predicts price_{t+1} >= (1 + threshold)*price_t
    - Exit next day at close(t+1).
    - No leverage, no shorting, 100% cash otherwise.
    Returns equity curve and stats.
    """
    df = backtest_df.copy().sort_index()

    df["Actual_next"] = df["Actual"].shift(-1)
    df["Signal"] = np.where(
        df["Predicted"].shift(1) >= df["Actual"] * (1 + threshold_pct / 100.0),
        1,
        0,
    )

    # Only enter when we know next day's actual
    df = df.dropna(subset=["Actual_next"])

    df["Return"] = df["Signal"] * (df["Actual_next"] - df["Actual"]) / df["Actual"]
    df["Equity"] = (1 + df["Return"]).cumprod()

    total_return = (df["Equity"].iloc[-1] - 1) * 100
    win_rate = (df["Return"] > 0).mean() * 100 if len(df) > 0 else 0.0

    stats = {
        "total_return_pct": total_return,
        "win_rate_pct": win_rate,
        "trades": int(df["Signal"].sum()),
    }
    return df, stats


with tabs[1]:
    st.header(f"📈 Charts & Prediction — {ticker if ticker else 'No Ticker'}")

    if ticker:
        # --------------------------
        # BASE CHART & NEXT-DAY DELTA
        # --------------------------
        latest_price, predicted_price = render_price_chart(ticker, timeframe)

        if latest_price is not None and predicted_price is not None:
            st.metric(
                label="Price Delta (Next Day Prediction)",
                value=f"${predicted_price - latest_price:.2f}",
            )

        st.markdown("---")

        # --------------------------
        # SHORT-TERM (7D) AI SUMMARY
        # --------------------------
        short_term_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        def get_short_term_summary(ticker: str, future_df: pd.DataFrame) -> str:
            """Generate a concise short-term outlook using the LLM."""
            prompt = f"""
            You are a short-term trading analyst.

            Stock: {ticker}
            Forecast Data:
            {future_df.to_string()}

            Provide ONLY:
            - Expected price move next 5–10 days
            - Support & resistance levels
            - Risk factors (max 3 bullets)
            - Suggested short-term action: Buy / Hold / Take Profit / Risky
            Keep output under 140 words.
            """

            res = short_term_llm.invoke(prompt)
            return res.content if hasattr(res, "content") else str(res)

        future_df_short = predict_future_prices(
            ticker, TIMEFRAME_MAPPING[timeframe], days_ahead=7
        )
        last_close_short = get_latest_price(ticker)

        if future_df_short is not None:
            st.subheader("📌 Short-Term Trading Insight (AI-Powered)")

            signal_info_short = generate_trading_signal(last_close_short, future_df_short)
            exp_ret = signal_info_short["expected_return_pct"]
            tgt_price = signal_info_short["target_price"]

            colA, colB, colC = st.columns(3)
            colA.metric("Signal", signal_info_short["signal"])
            colB.metric("Expected Return (7d)", f"{exp_ret:.2f}%")
            colC.metric("Target Price", f"${tgt_price:.2f}")

            summary = get_short_term_summary(ticker, future_df_short)
            st.info(summary)
        else:
            st.info("Not enough data for short-term AI outlook.")

        st.markdown("---")

        # --------------------------
        # BACKTEST (Last 30 Days) + Δ + DIR ACC + SIM
        # --------------------------
        backtest_df = predict_backtest_prices(
            ticker, TIMEFRAME_MAPPING[timeframe], days_back=30
        )

        if backtest_df is not None and not backtest_df.empty:
            # Basic deltas
            backtest_df["Delta"] = backtest_df["Predicted"] - backtest_df["Actual"]
            backtest_df["Delta_Pct"] = (
                backtest_df["Delta"] / backtest_df["Actual"]
            ) * 100

            # Directional accuracy (up/down correctly predicted)
            tmp = backtest_df.copy().sort_index()
            tmp["Actual_Change"] = tmp["Actual"].diff()
            tmp["Pred_Change"] = tmp["Predicted"].diff()
            tmp = tmp.dropna()
            dir_acc = (
                (np.sign(tmp["Actual_Change"]) == np.sign(tmp["Pred_Change"]))
                .mean()
                * 100
                if len(tmp) > 0
                else 0.0
            )

            st.subheader("📉 Prediction Error (Delta)")

            def delta_color(val):
                return "color: #28a745" if val >= 0 else "color: #dc3545"

            st.dataframe(
                backtest_df[["Actual", "Predicted", "Delta", "Delta_Pct"]]
                .style.format(
                    {
                        "Actual": "${:.2f}",
                        "Predicted": "${:.2f}",
                        "Delta": "${:.2f}",
                        "Delta_Pct": "{:.2f}%",
                    }
                )
                .applymap(delta_color, subset=["Delta", "Delta_Pct"])
            )

            # Delta bar chart
            fig_delta = go.Figure()
            fig_delta.add_trace(
                go.Bar(
                    x=backtest_df.index,
                    y=backtest_df["Delta"],
                    marker_color=[
                        "#28a745" if v >= 0 else "#dc3545"
                        for v in backtest_df["Delta"]
                    ],
                    name="Prediction Error (Δ)",
                )
            )
            fig_delta.update_layout(
                title="📉 Prediction Delta (Predicted − Actual)",
                xaxis_title="Date",
                yaxis_title="Δ Price (USD)",
                template=plotly_theme,
            )
            st.plotly_chart(fig_delta, use_container_width=True)

            # Actual vs Predicted chart
            st.subheader("📊 Backtest: Actual vs Predicted (Last 30 Days)")
            fig_bt = go.Figure()
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Actual"],
                    mode="lines+markers",
                    name="Actual",
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Predicted"],
                    mode="lines+markers",
                    name="Predicted",
                )
            )

            # Simple "confidence cone" using residual std dev
            residual_std = backtest_df["Delta"].std()
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Predicted"] + 2 * residual_std,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_bt.add_trace(
                go.Scatter(
                    x=backtest_df.index,
                    y=backtest_df["Predicted"] - 2 * residual_std,
                    mode="lines",
                    fill="tonexty",
                    name="±2σ band",
                    line=dict(width=0),
                )
            )

            fig_bt.update_layout(
                title="Actual vs Model Predicted (with ±2σ Band)",
                template=plotly_theme,
                yaxis_title="Price (USD)",
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            metrics = compute_backtest_metrics(backtest_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{metrics['rmse']:.2f}")
            c2.metric("MAE", f"{metrics['mae']:.2f}")
            c3.metric("MAPE", f"{metrics['mape']:.2f}%")
            c4.metric("Directional Accuracy", f"{dir_acc:.1f}%")

            st.markdown("---")

            # --------------------------
            # SIMPLE TRADING SIMULATION
            # --------------------------
            st.subheader("📈 Simple Long-Only Strategy Backtest")

            thr = st.slider(
                "Entry Threshold (predicted upside %) to trigger a trade",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.25,
                key="sim_threshold",
            )

            sim_df, sim_stats = run_trading_sim(backtest_df, threshold_pct=thr)

            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=sim_df.index,
                    y=sim_df["Equity"],
                    mode="lines",
                    name="Equity Curve",
                )
            )
            fig_eq.update_layout(
                title="Equity Curve — Simple Strategy",
                yaxis_title="Portfolio Value (Start = 1.0)",
                template=plotly_theme,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            colS1, colS2, colS3 = st.columns(3)
            colS1.metric("Total Return", f"{sim_stats['total_return_pct']:.2f}%")
            colS2.metric("Win Rate", f"{sim_stats['win_rate_pct']:.1f}%")
            colS3.metric("Trades", f"{sim_stats['trades']}")

        else:
            st.info("Not enough data for backtest evaluation and trading simulation.")

        st.markdown("---")

        # --------------------------
        # FUTURE FORECAST (Next 10 Days)
        # --------------------------
        future_df_full = predict_future_prices(
            ticker, TIMEFRAME_MAPPING[timeframe], days_ahead=10
        )
        last_close_full = get_latest_price(ticker)

        if future_df_full is not None:
            st.subheader("🔮 Future Price Forecast (Next 10 Business Days)")

            fig_fut = go.Figure()
            fig_fut.add_trace(
                go.Scatter(
                    x=future_df_full.index,
                    y=future_df_full["Predicted"],
                    mode="lines+markers",
                    name="Predicted",
                )
            )
            fig_fut.add_trace(
                go.Scatter(
                    x=future_df_full.index,
                    y=future_df_full["Upper_CI"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_fut.add_trace(
                go.Scatter(
                    x=future_df_full.index,
                    y=future_df_full["Lower_CI"],
                    mode="lines",
                    fill="tonexty",
                    name="95% CI",
                    line=dict(width=0),
                )
            )
            fig_fut.update_layout(
                template=plotly_theme,
                yaxis_title="Price (USD)",
                title="Model Forecast — Next 10 Business Days",
            )
            st.plotly_chart(fig_fut, use_container_width=True)

            signal_info_full = generate_trading_signal(last_close_full, future_df_full)
            st.metric(
                label=f"Trading Signal ({ticker})",
                value=signal_info_full["signal"],
                help=(
                    f"Expected return: "
                    f"{signal_info_full['expected_return_pct']:.2f}% → "
                    f"target ≈ ${signal_info_full['target_price']:.2f}"
                ),
            )
        else:
            st.info("Not enough data for 10-day forecast.")
    else:
        st.info("Please enter a ticker in the sidebar.")

# -----------------------------------------------------------------------------------
# 📊 Technical Indicators Tab
# -----------------------------------------------------------------------------------
import ta

with tabs[2]:
    st.header(f"📊 Technical Indicators — {ticker if ticker else 'No Ticker'}")

    if not ticker:
        st.info("Enter a ticker in the sidebar.")
        st.stop()

    period = TIMEFRAME_MAPPING[timeframe]
    df = get_stock_data(ticker, period)

    if df.empty:
        st.warning("No price data available.")
        st.stop()

    # Compute indicators
    df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA50"] = ta.trend.sma_indicator(df["Close"], window=50)

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    rsi = ta.momentum.RSIIndicator(df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    macd = ta.trend.MACD(
        df["Close"], window_slow=26, window_fast=12, window_sign=9
    )
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # UI Toggles for Indicators
    st.sidebar.subheader("📌 Technical Overlay Options")
    show_sma = st.sidebar.checkbox("Show SMA (20 & 50)", value=True)
    show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    show_macd = st.sidebar.checkbox("Show MACD", value=True)

    # Price + indicators
    fig_ti = go.Figure()

    fig_ti.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
    ))

    if show_sma:
        fig_ti.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"], name="SMA20", line=dict(width=2)
        ))
        fig_ti.add_trace(go.Scatter(
            x=df.index, y=df["SMA50"], name="SMA50", line=dict(width=2)
        ))

    if show_bbands:
        fig_ti.add_trace(go.Scatter(
            x=df.index, y=df["BB_high"], name="BB High", line=dict(width=1, dash="dash")
        ))
        fig_ti.add_trace(go.Scatter(
            x=df.index, y=df["BB_low"], name="BB Low", line=dict(width=1, dash="dash")
        ))

    fig_ti.update_layout(
        title=f"{ticker} — Price & Trend Indicators",
        template=plotly_theme,
        yaxis_title="Price (USD)",
        height=500,
    )
    st.plotly_chart(fig_ti, use_container_width=True)

    # RSI Chart
    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], mode="lines", name="RSI"
        ))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        fig_rsi.update_layout(
            title="Relative Strength Index (RSI)",
            template=plotly_theme,
            yaxis_title="RSI Value",
            height=250,
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD Chart
    if show_macd:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD"
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal", line=dict(width=1, dash="dash")
        ))
        fig_macd.update_layout(
            title="MACD & Signal Line",
            template=plotly_theme,
            height=250,
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    # --------------------------
    # 🚦 One-Click Trade Idea Summary
    # --------------------------
    st.markdown("---")
    st.subheader("🚦 One-Click Trade Idea")

    try:
        # Indicator scoring system
        score = 0
        total = 0

        # SMA Trend
        if show_sma:
            total += 1
            if df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]:
                score += 1
                sma_signal = "Bullish (SMA20 > SMA50)"
            else:
                sma_signal = "Bearish (SMA20 < SMA50)"

        # MACD
        if show_macd:
            total += 1
            if df["MACD"].iloc[-1] > df["MACD_signal"].iloc[-1]:
                score += 1
                macd_signal = "Bullish crossover"
            else:
                macd_signal = "Bearish crossover"

        # RSI
        if show_rsi:
            total += 1
            if df["RSI"].iloc[-1] < 30:
                score += 1
                rsi_signal = "Oversold (Bullish)"
            elif df["RSI"].iloc[-1] > 70:
                rsi_signal = "Overbought (Bearish)"
            else:
                score += 0.5
                rsi_signal = "Neutral"

        # Bollinger Bands
        if show_bbands:
            total += 1
            price = df["Close"].iloc[-1]
            if price <= df["BB_low"].iloc[-1]:
                score += 1
                bb_signal = "Reversal zone (Bullish)"
            elif price >= df["BB_high"].iloc[-1]:
                bb_signal = "Breakout (Bearish)"
            else:
                score += 0.5
                bb_signal = "Neutral"

        # Final Scoring
        if total > 0:
            confidence = (score / total) * 100
        else:
            confidence = 50

        # Trade bias
        if confidence >= 65:
            trade_signal = "BUY 🟢"
        elif confidence <= 40:
            trade_signal = "SELL 🔴"
        else:
            trade_signal = "HOLD 🟡"

        # Risk/Reward
        entry = df["Close"].iloc[-1]
        stop_loss = round(entry * 0.95, 2)
        take_profit = round(entry * 1.07, 2)

        st.metric("Signal", trade_signal)
        st.metric("Confidence", f"{confidence:.1f}%")

        # Details
        st.write(f"""
        **Indicators:**
        - 📊 SMA: {sma_signal if show_sma else "(disabled)"}
        - 📈 MACD: {macd_signal if show_macd else "(disabled)"}
        - 🔁 RSI: {rsi_signal if show_rsi else "(disabled)"}
        - 🎯 Bollinger: {bb_signal if show_bbands else "(disabled)"}

        **Suggested Trade Plan**
        - 🎯 Entry: **${entry:.2f}**
        - 💵 Take Profit: **${take_profit}**
        - 🛑 Stop Loss: **${stop_loss}**
        """)
    except Exception as e:
        st.warning(f"Trade idea unavailable: {e}")

# -------------------------------------------------------------
# 🧠 Analyst AI Tab (Meta-Agent)
# -------------------------------------------------------------
with tabs[3]:
    st.header(f"Meta-Agent Analyst — {ticker if ticker else 'No Ticker'}")

    analyst_question = st.text_area(
        "Ask the AI Analyst anything (e.g. 'Is this a buy?', 'What are key risks?')",
        "",
        height=100,
    )

    if st.button("Run Meta-Agent Analysis", key="meta_agent_run"):
        if not ticker:
            st.error("Please enter a ticker in the sidebar.")
        elif not analyst_question.strip():
            st.error("Please enter a question for the analyst.")
        else:
            with st.spinner("AI Analyst thinking…"):
                answer = ask_meta_agent(meta_agent, ticker, analyst_question)
            st.success("Analysis complete:")
            st.write(answer)

            st.session_state.responses.append(
                {
                    "ticker": ticker,
                    "analysis_type": "Meta-Agent",
                    "response": answer,
                }
            )

# -------------------------------------------------------------
# 💬 Chat & History Tab
# -------------------------------------------------------------
with tabs[4]:
    st.header("Chat with AI Analyst")

    # Show past conversation
    for msg in st.session_state.chat_history:
        role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {msg['content']}")

    st.markdown("---")

    col_input, col_voice = st.columns([3, 1])

    with col_input:
        user_chat_input = st.text_input(
            "Type your question:",
            key="chat_input",
            placeholder="e.g. What are the long-term prospects for this stock?",
        )
        send_chat = st.button("Send", key="send_chat")

    with col_voice:
        if mic_recorder is not None:
            st.write("Voice Input:")
            audio = mic_recorder(
                start_prompt="🎙️ Record",
                stop_prompt="⏹️ Stop",
                key="voice_recorder",
            )
            if audio:
                st.audio(audio["bytes"])
                st.info(
                    "Voice captured. For now, transcribe and paste your question in the text box.\n"
                    "You can integrate Whisper/OpenAI STT here if desired."
                )
        else:
            st.write("Voice input not available.\nInstall `streamlit-mic-recorder` to enable.")

    if send_chat and user_chat_input.strip():
        user_text = user_chat_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        with st.spinner("AI thinking…"):
            answer = ask_meta_agent(meta_agent, ticker or "UNKNOWN", user_text)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.responses.append(
            {
                "ticker": ticker,
                "analysis_type": "Chat",
                "response": answer,
            }
        )
        st.experimental_rerun()

    st.markdown("### Past Analyses")
    if st.session_state.responses:
        for entry in st.session_state.responses[::-1]:
            st.markdown(f"**Ticker:** {entry.get('ticker', 'N/A')}")
            st.markdown(f"**Type:** {entry.get('analysis_type', 'N/A')}")
            st.markdown(f"**Response:**\n{entry.get('response', 'No response')}")
            st.markdown("---")
    else:
        st.info("No past analyses yet.")

# -------------------------------------------------------------
# Single-Tool Analysis via Sidebar button (works across tabs)
# -------------------------------------------------------------
if run_single_analysis and ticker:
    with st.spinner(f"Running {analysis_type} for {ticker}…"):
        result = get_analysis(analysis_type, ticker, research_prompt)
    st.sidebar.success("Single analysis complete. See result below.")
    st.sidebar.write(result)

    st.session_state.responses.append(
        {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "response": result,
        }
    )
