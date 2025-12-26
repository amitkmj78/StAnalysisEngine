# app.py

from __future__ import annotations

import time
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta
import streamlit as st

# -------------------------------------------------------------
# Imports for services / agents
# -------------------------------------------------------------
from services.config import validate_environment
from services.llm_setup import init_llms
from services.data_service import TIMEFRAME_MAPPING, get_stock_data, get_latest_price
from services.prediction_service import (
    predict_next_price,
    predict_backtest_prices,
    compute_backtest_metrics,
    predict_future_prices,
    generate_trading_signal,
)
from services.analysis_service import get_analysis
from services.websocket_service import start_price_feed, is_ws_configured
from services.screener_service import (
    get_top_bullish_stocks,
    INDEX_MAP,
)
from services.ai_trade_idea_service import generate_trade_idea
from services.alert_service import alert_breakout
from services.trade_storage import save_trade_idea, load_trades, evaluate_trades
from services.positions_from_csv import positions_from_activity_csv
from services.portfolio_strategy import (
    build_robinhood_strategies,
    summarize_portfolio,
)
from services.manual_positions import build_manual_positions
from Agent.meta_agent import build_agent, ask_meta_agent
from langchain_openai import ChatOpenAI
from Agent.meta_agent import get_debug_logs
# -------------------------------------------------------------
# 🛡️ Risk Settings (Sidebar)
# -------------------------------------------------------------
st.set_page_config(page_title="AI Stock Intelligence Dashboard", layout="wide")

st.sidebar.subheader("🛡️ Risk Settings")

# Initialize defaults in session_state
if "risk_profile" not in st.session_state:
    st.session_state["risk_profile"] = "Balanced"

if "risk_factor" not in st.session_state:
    st.session_state["risk_factor"] = 5

risk_profile = st.sidebar.selectbox(
    "Risk Tolerance Level:",
    ["Aggressive", "Balanced", "Conservative"],
    index=["Aggressive", "Balanced", "Conservative"].index(
        st.session_state["risk_profile"]
    ),
    key="risk_profile_select",
)

risk_factor = st.sidebar.slider(
    "Fine Tune Risk Level:",
    min_value=1,
    max_value=10,
    value=st.session_state["risk_factor"],
    step=1,
    key="risk_factor_slider",
    help=(
        "Lower = safer & tighter stops, "
        "Higher = bigger targets & more volatility tolerance."
    ),
)

st.session_state["risk_profile"] = risk_profile
st.session_state["risk_factor"] = risk_factor

# -------------------------------------------------------------
# 🎨 Theme Settings (Sidebar)
# -------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark Mode"

st.sidebar.header("⚙️ UI Settings")

theme = st.sidebar.radio(
    "Theme",
    ("Dark Mode", "Light Mode"),
    index=0 if st.session_state["theme"] == "Dark Mode" else 1,
)

st.session_state["theme"] = theme

st.sidebar.markdown(
    f"**Profile:** {st.session_state['risk_profile']}  \n"
    f"**Risk Factor:** {st.session_state['risk_factor']}/10"
)

# -------------------------------------------------------------
# Optional: Voice Input Setup
# -------------------------------------------------------------
try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None

# -------------------------------------------------------------
# Theme for Plotly charts
# -------------------------------------------------------------
PLOTLY_THEME = "plotly_dark" if st.session_state["theme"] == "Dark Mode" else "plotly_white"

# -------------------------------------------------------------
# Environment validation
# -------------------------------------------------------------
validate_environment()

# -------------------------------------------------------------
# LLM Setup & Meta-Agent
# -------------------------------------------------------------
# llm_openai, llm_groq, llm_labels = init_llms()

llm_openai, llm_groq, llm_ollama, llm_labels = init_llms()

if not llm_labels:
    st.error("❌ No working LLM loaded.")
    st.stop()

# Model selection dropdown
select_llm = st.sidebar.selectbox("Select LLM Model", llm_labels)

if select_llm.startswith("OpenAI"):
    llm = llm_openai
elif select_llm.startswith("Groq"):
    llm = llm_groq
elif select_llm.startswith("Local"):
    llm = llm_ollama
else:
    llm = None

# Safety: ensure llm is not None
if llm is None:
    st.error("❌ Selected model failed to initialize.")
    st.stop()

# Build/Rebuild Meta-Agent when model changes
if "meta_agent" not in st.session_state or \
   st.session_state.get("meta_agent_model") != select_llm:

    st.session_state.meta_agent = build_agent(llm)
    st.session_state.meta_agent_model = select_llm
    st.toast(f"🔄 Meta-agent rebuilt using {select_llm}")

meta_agent = st.session_state.meta_agent

# -------------------------------------------------------------
# Sidebar Controls for Ticker etc.
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
    "Prediction Days (label only here)",
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
# Session State for Chat / History
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "responses" not in st.session_state:
    st.session_state.responses = []

# -------------------------------------------------------------
# Tabs Layout
# -------------------------------------------------------------
tabs = st.tabs(
    [
        "🏠 Overview",              # 0
        "📈 Charts & Prediction",   # 1
        "📊 Technical Indicators",  # 2
        "📊 Bullish Screener",      # 3
        "⏱ Live Price Feed",        # 4
        "🧠 Analyst AI",            # 5
        "💬 Chat & History",        # 6
        "🔥 Top 10 Stocks",         # 7
        "📘 Trade Journal",         # 8
        "📘 Portfolio Strategies (Robinhood)",  # 9
        "📝 Manual Portfolio Strategies",       # 10
    ]
)

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
        template=PLOTLY_THEME,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    return latest_price, predicted_price

# ===================================================================
# 🏠 Overview Tab
# ===================================================================
with tabs[0]:
    st.title("📊 AI Stock Intelligence Dashboard")
    
    st.text_area("Debug Output", get_debug_logs(), height=400)
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
            snapshot = get_analysis("Basic Info", ticker, None)
            st.write(snapshot)
        except Exception as e:
            st.warning(f"Unable to load basic info right now: {e}")
    else:
        st.info("Enter a ticker in the sidebar to get started.")

# ===================================================================
# 📈 Charts & Prediction Tab
# ===================================================================
def run_trading_sim(backtest_df: pd.DataFrame, threshold_pct: float = 0.0):
    """
    Very simple strategy:
    - Go long at close(t) if model predicts price_{t+1} >= (1 + threshold)*price_t
    - Exit next day at close(t+1).
    """
    df = backtest_df.copy().sort_index()

    df["Actual_next"] = df["Actual"].shift(-1)
    df["Signal"] = np.where(
        df["Predicted"].shift(1) >= df["Actual"] * (1 + threshold_pct / 100.0),
        1,
        0,
    )

    df = df.dropna(subset=["Actual_next"])
    df["Return"] = df["Signal"] * (df["Actual_next"] - df["Actual"]) / df["Actual"]
    df["Equity"] = (1 + df["Return"]).cumprod()

    if df.empty:
        stats = {"total_return_pct": 0.0, "win_rate_pct": 0.0, "trades": 0}
    else:
        total_return = (df["Equity"].iloc[-1] - 1) * 100
        win_rate = (df["Return"] > 0).mean() * 100
        stats = {
            "total_return_pct": total_return,
            "win_rate_pct": win_rate,
            "trades": int(df["Signal"].sum()),
        }
    return df, stats


with tabs[1]:
    st.header(f"📈 Charts & Prediction — {ticker if ticker else 'No Ticker'}")

    if ticker:
        latest_price, predicted_price = render_price_chart(ticker, timeframe)

        if latest_price is not None and predicted_price is not None:
            st.metric(
                label="Price Delta (Next Day Prediction)",
                value=f"${predicted_price - latest_price:.2f}",
            )

        st.markdown("---")

        # Short-term (7D) AI summary
        short_term_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        def get_short_term_summary(ticker: str, future_df: pd.DataFrame) -> str:
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

        if future_df_short is not None and last_close_short is not None:
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

        # Backtest (last 30 days)
        backtest_df = predict_backtest_prices(
            ticker, TIMEFRAME_MAPPING[timeframe], days_back=30
        )

        if backtest_df is not None and not backtest_df.empty:
            backtest_df["Delta"] = backtest_df["Predicted"] - backtest_df["Actual"]
            backtest_df["Delta_Pct"] = (
                backtest_df["Delta"] / backtest_df["Actual"] * 100
            )

            tmp = backtest_df.copy().sort_index()
            tmp["Actual_Change"] = tmp["Actual"].diff()
            tmp["Pred_Change"] = tmp["Predicted"].diff()
            tmp = tmp.dropna()
            if len(tmp) > 0:
                dir_acc = (
                    (np.sign(tmp["Actual_Change"]) == np.sign(tmp["Pred_Change"]))
                    .mean()
                    * 100
                )
            else:
                dir_acc = 0.0

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
                template=PLOTLY_THEME,
            )
            st.plotly_chart(fig_delta, use_container_width=True)

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
                template=PLOTLY_THEME,
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

            # Simple trading simulation
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
                template=PLOTLY_THEME,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            colS1, colS2, colS3 = st.columns(3)
            colS1.metric("Total Return", f"{sim_stats['total_return_pct']:.2f}%")
            colS2.metric("Win Rate", f"{sim_stats['win_rate_pct']:.1f}%")
            colS3.metric("Trades", f"{sim_stats['trades']}")
        else:
            st.info("Not enough data for backtest evaluation and trading simulation.")

        st.markdown("---")

        # Future forecast (next 10 days)
        future_df_full = predict_future_prices(
            ticker, TIMEFRAME_MAPPING[timeframe], days_ahead=10
        )
        last_close_full = get_latest_price(ticker)

        if future_df_full is not None and last_close_full is not None:
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
                template=PLOTLY_THEME,
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

# ===================================================================
# 📊 Technical Indicators Tab
# ===================================================================
with tabs[2]:
    st.header(f"📊 Technical Indicators — {ticker if ticker else 'No Ticker'}")

    if not ticker:
        st.info("Enter a ticker in the sidebar.")
    else:
        period = TIMEFRAME_MAPPING[timeframe]
        df = get_stock_data(ticker, period)

        if df.empty:
            st.warning("No price data available.")
        else:
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

            st.sidebar.subheader("📌 Technical Overlay Options")
            show_sma = st.sidebar.checkbox("Show SMA (20 & 50)", value=True)
            show_bbands = st.sidebar.checkbox("Show Bollinger Bands", value=True)
            show_rsi = st.sidebar.checkbox("Show RSI", value=True)
            show_macd = st.sidebar.checkbox("Show MACD", value=True)

            fig_ti = go.Figure()

            fig_ti.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                )
            )

            if show_sma:
                fig_ti.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["SMA20"],
                        name="SMA20",
                        line=dict(width=2),
                    )
                )
                fig_ti.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["SMA50"],
                        name="SMA50",
                        line=dict(width=2),
                    )
                )

            if show_bbands:
                fig_ti.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_high"],
                        name="BB High",
                        line=dict(width=1, dash="dash"),
                    )
                )
                fig_ti.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_low"],
                        name="BB Low",
                        line=dict(width=1, dash="dash"),
                    )
                )

            fig_ti.update_layout(
                title=f"{ticker} — Price & Trend Indicators",
                template=PLOTLY_THEME,
                yaxis_title="Price (USD)",
                height=500,
            )
            st.plotly_chart(fig_ti, use_container_width=True)

            if show_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=df.index, y=df["RSI"], mode="lines", name="RSI"
                    )
                )
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
                fig_rsi.update_layout(
                    title="Relative Strength Index (RSI)",
                    template=PLOTLY_THEME,
                    yaxis_title="RSI Value",
                    height=250,
                )
                st.plotly_chart(fig_rsi, use_container_width=True)

            if show_macd:
                fig_macd = go.Figure()
                fig_macd.add_trace(
                    go.Scatter(x=df.index, y=df["MACD"], name="MACD")
                )
                fig_macd.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MACD_signal"],
                        name="Signal",
                        line=dict(width=1, dash="dash"),
                    )
                )
                fig_macd.update_layout(
                    title="MACD & Signal Line",
                    template=PLOTLY_THEME,
                    height=250,
                )
                st.plotly_chart(fig_macd, use_container_width=True)

            st.markdown("---")
            st.subheader("🚦 One-Click Trade Idea")

            try:
                score = 0
                total = 0

                if show_sma:
                    total += 1
                    if df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]:
                        score += 1
                        sma_signal = "Bullish (SMA20 > SMA50)"
                    else:
                        sma_signal = "Bearish (SMA20 < SMA50)"
                else:
                    sma_signal = "(disabled)"

                if show_macd:
                    total += 1
                    if df["MACD"].iloc[-1] > df["MACD_signal"].iloc[-1]:
                        score += 1
                        macd_signal = "Bullish crossover"
                    else:
                        macd_signal = "Bearish crossover"
                else:
                    macd_signal = "(disabled)"

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
                else:
                    rsi_signal = "(disabled)"

                if show_bbands:
                    total += 1
                    price_last = df["Close"].iloc[-1]
                    if price_last <= df["BB_low"].iloc[-1]:
                        score += 1
                        bb_signal = "Reversal zone (Bullish)"
                    elif price_last >= df["BB_high"].iloc[-1]:
                        bb_signal = "Breakout (Bearish)"
                    else:
                        score += 0.5
                        bb_signal = "Neutral"
                else:
                    bb_signal = "(disabled)"

                confidence = (score / total) * 100 if total > 0 else 50

                if confidence >= 65:
                    trade_signal = "BUY 🟢"
                elif confidence <= 40:
                    trade_signal = "SELL 🔴"
                else:
                    trade_signal = "HOLD 🟡"

                entry = df["Close"].iloc[-1]
                stop_loss = round(entry * 0.95, 2)
                take_profit = round(entry * 1.07, 2)

                st.metric("Signal", trade_signal)
                st.metric("Confidence", f"{confidence:.1f}%")

                st.write(
                    f"""
                **Indicators:**
                - 📊 SMA: {sma_signal}
                - 📈 MACD: {macd_signal}
                - 🔁 RSI: {rsi_signal}
                - 🎯 Bollinger: {bb_signal}

                **Suggested Trade Plan**
                - 🎯 Entry: **${entry:.2f}**
                - 💵 Take Profit: **${take_profit}**
                - 🛑 Stop Loss: **${stop_loss}**
                """
                )
            except Exception as e:
                st.warning(f"Trade idea unavailable: {e}")

# ===================================================================
# 📊 Bullish Screener Tab
# ===================================================================
with tabs[3]:
    st.header("📊 Top Bullish Opportunities")

    index_key = st.selectbox(
        "Market Universe:",
        list(INDEX_MAP.keys()),
        index=0,
        key="bullish_index_key",
    )

    period_label = st.selectbox(
        "Performance Lookback (primary sort):",
        ["3mo", "6mo", "1y"],
        index=2,
        key="bullish_period",
    )

    top_n = st.slider(
        "Number of Stocks:",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="bullish_topn",
    )

    if st.button("Run Screener 🚀", key="run_bullish_screener"):
        with st.spinner("Scanning markets for bullish setups…"):
            df = get_top_bullish_stocks(
                index_key=index_key,
                period=period_label,
                top_n=top_n,
            )

        if df.empty:
            st.warning("No bullish stocks found for this universe / period.")
        else:
            st.success(f"Found {len(df)} bullish candidates in **{index_key}**")
            st.dataframe(df, use_container_width=True)

            st.caption("👇 Click a ticker to load it in sidebar")

            for idx, row in df.iterrows():
                tk = row["Ticker"]
                col1, col2, col3 = st.columns([2, 3, 2])

                with col1:
                    st.subheader(f"{tk}")
                    st.write(row["Rating"])
                    st.write(f"Score: **{row['Score']}**")

                with col2:
                    import yfinance as yf
                    hist = yf.Ticker(tk).history(period="6mo", auto_adjust=True)
                    if not hist.empty:
                        st.line_chart(hist["Close"].tail(60))

                with col3:
                    if st.button(f"📥 Load {tk}", key=f"load_{tk}_{idx}"):
                        st.session_state["Stock Ticker"] = tk
                        # You can call st.rerun() here if desired

# ===================================================================
# ⏱ Live Price Feed Tab
# ===================================================================
with tabs[4]:
    st.header(f"Real-Time WebSocket Price — {ticker if ticker else 'No Ticker'}")

    if not is_ws_configured():
        st.warning(
            "Live feed is not configured. "
            "Set FINNHUB_API_KEY or WS_PRICE_FEED_URL in your .env, "
            "and install `websocket-client`."
        )
    elif not ticker:
        st.info("Enter a ticker in the sidebar to see live prices.")
    else:
        if enable_live_feed:
            start_price_feed(ticker)
            live_price = st.session_state.get("live_price")

            if live_price is not None:
                st.metric(
                    label=f"Live Price for {ticker}",
                    value=f"${live_price:.4f}",
                )
            else:
                st.info("Connected. Waiting for first tick...")

            time.sleep(1)
            # You can re-enable st.rerun() if you want continuous updates
            # st.rerun()
        else:
            st.info("Enable the live feed in the sidebar to start streaming prices.")

# ===================================================================
# 🧠 Analyst AI Tab (Meta-Agent)
# ===================================================================
with tabs[5]:
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

# ===================================================================
# 💬 Chat & History Tab
# ===================================================================
with tabs[6]:
    st.header("Chat with AI Analyst")

    # Show history
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
            st.write(
                "Voice input not available.\nInstall `streamlit-mic-recorder` to enable."
            )

    # ✅ Only call meta-agent WHEN user clicks Send and entered text
    if send_chat and user_chat_input.strip():
        user_text = user_chat_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        if not ticker:
            answer = "⚠️ Please enter a ticker in the sidebar first."
        else:
            with st.spinner("AI thinking…"):
                answer = ask_meta_agent(meta_agent, ticker, user_text)

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

# ===================================================================
# 🔥 Top 10 Stocks Tab (Index-based Screener with Trade Saving)
# ===================================================================
with tabs[7]:
    st.header("🔥 Top Bullish Stocks (1-Year Momentum + AI + Analysts)")

    index_ui = st.selectbox("Select Market Index:", list(INDEX_MAP.keys()))
    top_n = st.slider("Number of Stocks:", 5, 50, 10)

    run_screen = st.button("Run Screener 🚀", key="run_top10_screener")

    if run_screen:
        from services.trade_storage import save_trade_idea

        with st.spinner("Scanning markets…"):
            df = get_top_bullish_stocks(
                index_key=index_ui,
                period="1y",
                top_n=top_n,
            )

        if df.empty:
            st.warning("No bullish stocks found today.")
        else:
            st.success(f"Top {len(df)} bullish names in **{index_ui}**")

            # ------------------------------------------
            # 🚀 AUTO-SAVE TRADE IDEAS DURING SCAN
            # ------------------------------------------
            auto_saved = 0

            for _, row in df.iterrows():
                tk = row["Ticker"]

                # Build context text
                context_text = (
                    f"1Y Return {row.get('Return_1Y_%', 'N/A')}%, "
                    f"6M Return {row.get('Return_6M_%', 'N/A')}%, "
                    f"Score {row.get('Score', 'N/A')}"
                )

                # Generate the idea
                idea_text, entry_low, entry_high, stop_loss, target = generate_trade_idea(
                    ticker=tk,
                    context=context_text,
                )

                # Auto-save directly (no button)
                save_trade_idea(
                    ticker=tk,
                    entry_low=entry_low,
                    entry_high=entry_high,
                    stop_loss=stop_loss,
                    target=target,
                    direction="LONG",
                    strategy_type="Index Screener (Auto)",
                    context=idea_text,
                    risk_profile=st.session_state.get("risk_profile", "Balanced"),
                    risk_factor=st.session_state.get("risk_factor", 5),
                )
                evaluate_trades()   # auto-run backtest after saving
                auto_saved += 1

                # UI display
                st.subheader(f"📌 {tk} — Saved Automatically ✔️")
                st.info(idea_text)

                # Sparkline
                import yfinance as yf
                hist = yf.Ticker(tk).history(period="6mo", auto_adjust=True)
                if not hist.empty:
                    st.line_chart(hist["Close"].tail(60))

                st.markdown("---")

            st.success(f"✅ Auto-saved {auto_saved} trade ideas into your Trade Journal!")

# ===================================================================
# 📘 Trade Journal Tab
# ===================================================================
with tabs[8]:
    st.header("📓 Trade Journal & Backtester")
    from services.portfolio_storage import load_table, delete_all_rows, delete_row_by_id

    # -----------------------------
    # Evaluate or Load Trades
    # -----------------------------
    if st.button("🔄 Evaluate All Trades", use_container_width=True):
        with st.spinner("Running backtest evaluation…"):
            df = evaluate_trades()
        st.success("Trade performance updated.")
    else:
        df = load_trades()

    st.subheader("⚠️ Danger Zone")

    if st.button("🗑️ Delete ALL Trades", use_container_width=True):
        st.warning("Are you sure you want to permanently delete ALL trades?")
    if st.button("YES, DELETE EVERYTHING"):
        delete_all_rows("portfolio_positions")   # your main journal table
        delete_all_rows("portfolio_strategies")
        st.success("All trades deleted.")
        st.stop()

    # -----------------------------
    # Additional Tables: Manual, Robinhood, Strategies
    # -----------------------------
    st.markdown("## 📁 Trade Sources Overview")
    from services.portfolio_storage import (
        save_portfolio_positions,
        save_portfolio_strategies,
        load_table,
    )
    manual_df = load_table("portfolio_positions")
    robinhood_df = load_table("portfolio_positions_rh")
    strategies_df = load_table("portfolio_strategies")

    # Manual Trades
    st.subheader("✍️ Manual Trades")
    if manual_df.empty:
        st.info("No manual trades recorded yet.")
    else:
        st.dataframe(manual_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Robinhood Trades
    st.subheader("📲 Robinhood Imported Trades")
    if robinhood_df.empty:
        st.info("No Robinhood trades imported. Upload CSV to begin.")
    else:
        st.dataframe(robinhood_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Strategy Trades
    st.subheader("🧠 Strategy-Based Trades")
    if strategies_df.empty:
        st.info("No strategy trades saved yet.")
    else:
        st.dataframe(strategies_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------
    # Main Journal View (Trade Ideas / Backtester)
    # -----------------------------
    if df.empty:
        st.info("No trades saved yet. Run the screener or manual trade idea tool.")
    else:
        for c in ["Created_At", "Entry_Date", "Exit_Date"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        st.subheader("🔎 Filter Trades")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        with col_f1:
            status_filter = st.selectbox(
                "Status",
                ["All", "OPEN", "ACTIVE", "TARGET_HIT", "STOP_HIT", "EXPIRED"],
            )

        with col_f2:
            ticker_filter = st.selectbox(
                "Ticker",
                ["All"] + sorted(df["Ticker"].unique().tolist()),
            )

        with col_f3:
            strat_filter = st.selectbox(
                "Strategy Type",
                ["All"] + sorted(df["Strategy_Type"].unique().tolist()),
            )

        with col_f4:
            dir_filter = st.selectbox(
                "Direction",
                ["All"] + sorted(df["Direction"].unique().tolist()),
            )

        filtered = df.copy()
        if status_filter != "All":
            filtered = filtered[filtered["Status"] == status_filter]
        if ticker_filter != "All":
            filtered = filtered[filtered["Ticker"] == ticker_filter]
        if strat_filter != "All":
            filtered = filtered[filtered["Strategy_Type"] == strat_filter]
        if dir_filter != "All":
            filtered = filtered[filtered["Direction"] == dir_filter]

        st.write(f"**Showing {len(filtered)} trades**")

        st.subheader("🗑️ Trade Journal (Delete Rows)")
        df_display = filtered.copy()
        df_display["Delete"] = False  # checkbox for delete

        edited = st.data_editor(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Delete": st.column_config.CheckboxColumn(
                    label="🗑️",
                    help="Select row to delete",
                )
            },
        )

        rows_to_delete = edited[edited["Delete"] == True]

        if not rows_to_delete.empty:
            if st.button("Delete Selected Trades", type="primary"):
                for _, row in rows_to_delete.iterrows():
                    trade_id = row["ID"]
                    delete_row_by_id("portfolio_strategies", "ID", trade_id)
                st.success(f"Deleted {len(rows_to_delete)} trade(s).")
                st.experimental_rerun()

        from services.portfolio_storage import list_tables, show_sqlite_table
        with st.expander("🗄️ View SQLite Database Tables"):
            st.subheader("SQLite Table Viewer")

            tables = list_tables()

            if not tables:
                st.warning("No tables found in trades.db.")
            else:
                table_name = st.selectbox("Select a table:", tables)

                if st.button("Show Table"):
                    df_sql = show_sqlite_table(table_name)
                    st.dataframe(df_sql, use_container_width=True)

        # -----------------------------
        # Performance Summary
        # -----------------------------
        st.subheader("📊 Performance Summary")
        closed = filtered[
            filtered["Status"].isin(["TARGET_HIT", "STOP_HIT", "EXPIRED"])
        ]

        if closed.empty:
            st.info("No closed trades yet. Summary will appear once trades complete.")
        else:
            total_return = closed["Realized_PnL_Pct"].mean()
            win_rate = (closed["Realized_PnL_Pct"] > 0).mean() * 100
            best = closed["Realized_PnL_Pct"].max()
            worst = closed["Realized_PnL_Pct"].min()
            profit_factor = (
                closed[closed["Realized_PnL_Pct"] > 0]["Realized_PnL_Pct"].sum()
                / abs(
                    closed[closed["Realized_PnL_Pct"] < 0]["Realized_PnL_Pct"].sum()
                )
                if any(closed["Realized_PnL_Pct"] < 0)
                else np.nan
            )

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Win Rate", f"{win_rate:.1f}%")
            col_s2.metric("Average Return", f"{total_return:.2f}%")
            col_s3.metric("Best Trade", f"{best:.2f}%" if pd.notna(best) else "N/A")
            col_s4.metric("Worst Trade", f"{worst:.2f}%" if pd.notna(worst) else "N/A")

            st.markdown("---")

            st.subheader("📈 Equity Curve (Realized returns)")
            closed_sorted = closed.sort_values("Exit_Date")
            closed_sorted["Equity"] = (1 + closed_sorted["Realized_PnL_Pct"] / 100).cumprod()
            st.line_chart(closed_sorted.set_index("Exit_Date")["Equity"])

            st.markdown("---")
            st.subheader("🏆 Top Performance Breakdown")
            col_b1, col_b2 = st.columns(2)

            with col_b1:
                st.write("**Best Tickers**")
                st.dataframe(
                    closed.groupby("Ticker")["Realized_PnL_Pct"]
                    .mean()
                    .sort_values(ascending=False)
                )

            with col_b2:
                st.write("**Best Strategies**")
                st.dataframe(
                    closed.groupby("Strategy_Type")["Realized_PnL_Pct"]
                    .mean()
                    .sort_values(ascending=False)
                )

            st.markdown("---")
            st.subheader("🔍 Trade Drill-Down")

            trade_ids = filtered["Trade_ID"].tolist()
            if trade_ids:
                selected_id = st.selectbox("Select Trade:", trade_ids)

                trade_row = filtered[filtered["Trade_ID"] == selected_id].iloc[0]
                st.write("### Trade Details")
                st.json(trade_row.to_dict())

                import yfinance as yf

                tk = trade_row["Ticker"]
                hist = yf.Ticker(tk).history(period="6mo")

                if not hist.empty:
                    st.subheader(f"📊 Price Chart: {tk}")
                    st.line_chart(hist["Close"])

# ===================================================================
# 📘 Portfolio Strategies (Robinhood)
# ===================================================================
with tabs[9]:
    st.header("📘 Portfolio Strategies from Robinhood Holdings")

    st.subheader("📂 Import Broker Activity & Rebuild Positions")

    uploaded = st.file_uploader("Upload activity CSV", type=["csv"])

    holdings = None

    if uploaded is not None:
        try:
            holdings = positions_from_activity_csv(uploaded)

            if "Net_Shares" in holdings.columns and "Shares" not in holdings.columns:
                holdings = holdings.rename(columns={"Net_Shares": "Shares"})

            if holdings.empty:
                st.warning("No open positions found (all trades fully closed).")
            else:
                st.success("Positions reconstructed from activity CSV:")
                st.dataframe(holdings, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

    if holdings is not None and not holdings.empty:
        risk_profile = st.session_state.get("risk_profile", "Balanced")
        risk_factor = st.session_state.get("risk_factor", 5)

        st.markdown(
            f"**Risk Profile:** `{risk_profile}`  |  **Risk Factor:** `{risk_factor}/10`"
        )
        st.markdown("---")
        from services.portfolio_storage import save_portfolio_positions, save_portfolio_strategies, load_table

        # Save reconstructed Robinhood positions
        save_portfolio_positions(holdings, source="Robinhood")

        with st.spinner("Building strategies based on your risk settings…"):
            strat_df = build_robinhood_strategies(
                holdings_df=holdings,
                risk_profile=risk_profile,
                risk_factor=risk_factor,
            )
        save_portfolio_strategies(
            strat_df,
            risk_profile=risk_profile,
            risk_factor=risk_factor,
        )

        if strat_df.empty:
            st.warning("⚠ No valid strategies generated.")
        else:
            summary = summarize_portfolio(strat_df)

            st.markdown(
                f"""
                ### 📈 Portfolio Summary
                - **Total Positions:** {summary['total_positions']}
                - **Portfolio Value:** `${summary['total_value']:.2f}`
                - **Unrealized PnL:** {summary['total_pnl_pct']:+.2f}%  
                """
            )
            st.markdown("---")

            for _, row in strat_df.iterrows():
                st.markdown(f"## 🧾 {row['Ticker']}")

                st.write(
                    f"Shares: **{row['Shares']}**  |  "
                    f"Avg Cost: `${row['Avg_Cost']:.2f}`  |  "
                    f"Current Price: `${row['Current_Price']:.2f}`  |  "
                    f"PnL: {row['Unrealized_PnL_%']:+.2f}%"
                )

                short_plan = row.get("Short_Term_Plan", "*No short-term plan generated*")
                long_plan = row.get("Long_Term_Plan", "*No long-term plan generated*")

                st.markdown("### ⚡ Short-Term Strategy")
                st.markdown(short_plan)

                st.markdown("### 🛡️ Long-Term Strategy")
                st.markdown(long_plan)

                st.markdown("---")

# ===================================================================
# 📝 Manual Portfolio Strategies Tab
# ===================================================================
with tabs[10]:
    st.header("📝 Manual Position Entry → Portfolio Strategies")

    st.subheader("Enter Your Positions Manually")

    num_rows = st.number_input(
        "Number of Positions", min_value=1, max_value=20, value=3
    )

    names = []
    tickers_list = []
    shares_list = []
    prices = []
    avg_costs = []
    total_returns = []

    st.markdown("### Enter Position Details")

    for i in range(num_rows):
        st.markdown(f"#### Position {i + 1}")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        names.append(col1.text_input(f"Name {i + 1}", ""))
        tickers_list.append(col2.text_input(f"Symbol {i + 1}", "").upper())
        shares_list.append(col3.number_input(f"Shares {i + 1}", min_value=0.0, step=1.0))
        prices.append(col4.number_input(f"Current Price {i + 1}", min_value=0.0, step=0.01))
        avg_costs.append(col5.number_input(f"Average Cost {i + 1}", min_value=0.0, step=0.01))
        total_returns.append(col6.text_input(f"Total Return % (optional) {i + 1}", ""))

    run_btn = st.button("Run Strategies 🚀")

    if run_btn:
        holdings = build_manual_positions(
            names,
            tickers_list,
            shares_list,
            prices,
            avg_costs,
            total_returns,
        )
        from services.portfolio_storage import save_portfolio_positions, save_portfolio_strategies, load_table

        save_portfolio_positions(holdings, source="Manual")

        if holdings.empty:
            st.error("No valid positions entered.")
        else:
            st.success("Manual positions loaded:")
            st.dataframe(holdings, use_container_width=True)

            risk_profile = st.session_state.get("risk_profile", "Balanced")
            risk_factor = st.session_state.get("risk_factor", 5)

            with st.spinner("Building strategies…"):
                strat_df = build_robinhood_strategies(
                    holdings_df=holdings,
                    risk_profile=risk_profile,
                    risk_factor=risk_factor,
                )
            save_portfolio_strategies(
                strat_df,
                risk_profile=risk_profile,
                risk_factor=risk_factor,
            )
            summary = summarize_portfolio(strat_df)

            st.markdown(
                f"""
                ### 📈 Portfolio Summary
                - **Total Positions:** {summary['total_positions']}
                - **Portfolio Value:** `${summary['total_value']:.2f}`
                - **Unrealized PnL:** {summary['total_pnl_pct']:+.2f}%  
                """
            )

            st.markdown("---")

            for _, row in strat_df.iterrows():
                st.markdown(f"## 🧾 {row['Ticker']}")

                st.write(
                    f"Shares: **{row['Shares']}**  |  "
                    f"Avg Cost: `${row['Avg_Cost']:.2f}`  |  "
                    f"Current Price: `${row['Current_Price']:.2f}`  |  "
                    f"PnL: {row['Unrealized_PnL_%']:+.2f}%"
                )

                st.markdown("### ⚡ Short-Term Strategy")
                st.markdown(row.get("Short_Term_Plan", "*No plan generated*"))

                st.markdown("### 🛡 Long-Term Strategy")
                st.markdown(row.get("Long_Term_Plan", "*No plan generated*"))

                st.markdown("---")

# ===================================================================
# Single-Tool Analysis via Sidebar button (works across tabs)
# ===================================================================
if run_single_analysis and ticker:
    with st.spinner(f"Running {analysis_type} for {ticker}…"):
        result = get_analysis(analysis_type, ticker, research_prompt or None)
    st.sidebar.write("🧠 Using Agent Tools…")
    st.sidebar.success("Single analysis complete. See result below.")
    st.sidebar.write(result)

    st.session_state.responses.append(
        {
            "ticker": ticker,
            "analysis_type": analysis_type,
            "response": result,
        }
    )
