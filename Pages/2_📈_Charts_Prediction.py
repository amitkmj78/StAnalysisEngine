import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
import numpy as np
from services.data_service import TIMEFRAME_MAPPING, get_latest_price
from services.prediction_service import (
    predict_backtest_prices,
    predict_next_price,
    predict_future_prices,
    generate_trading_signal,
    compute_backtest_metrics,
)

# Your helper functions moved from main app
from services.your_chart_helpers import render_price_chart, run_trading_sim


# ===================================================================
# PAGE TITLE
# ===================================================================
#st.title("📈 Charts & Prediction")

ticker = st.session_state.get("ticker")
timeframe = st.session_state.get("timeframe")
PLOTLY_THEME = "plotly_dark" if st.session_state.get("theme") == "Dark Mode" else "plotly_white"


# ===================================================================
# VALIDATION
# ===================================================================
if not ticker:
    st.info("Please enter a ticker in the sidebar.")
    st.stop()


st.header(f"📈 Charts & Prediction — {ticker}")


# ===================================================================
# PRICE CHART + NEXT-DAY PREDICTION
# ===================================================================
latest_price, predicted_price = render_price_chart(ticker, timeframe)

if latest_price is not None and predicted_price is not None:
    st.metric(
        label="Price Delta (Next Day Prediction)",
        value=f"${predicted_price - latest_price:.2f}",
    )

st.markdown("---")


# ===================================================================
# SHORT-TERM (7 DAY) AI MARKET INSIGHT
# ===================================================================
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


future_df_short = predict_future_prices(ticker, TIMEFRAME_MAPPING[timeframe], days_ahead=7)
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


# ===================================================================
# BACKTEST (LAST 30 DAYS)
# ===================================================================
backtest_df = predict_backtest_prices(ticker, TIMEFRAME_MAPPING[timeframe], days_back=30)

if backtest_df is not None and not backtest_df.empty:

    # Compute delta
    backtest_df["Delta"] = backtest_df["Predicted"] - backtest_df["Actual"]
    backtest_df["Delta_Pct"] = (backtest_df["Delta"] / backtest_df["Actual"]) * 100

    # Direction accuracy
    tmp = backtest_df.copy().sort_index()
    tmp["Actual_Change"] = tmp["Actual"].diff()
    tmp["Pred_Change"] = tmp["Predicted"].diff()
    tmp = tmp.dropna()

    dir_acc = (
        (np.sign(tmp["Actual_Change"]) == np.sign(tmp["Pred_Change"])).mean() * 100
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
                "#28a745" if v >= 0 else "#dc3545" for v in backtest_df["Delta"]
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

    # Backtest plot
    st.subheader("📊 Backtest: Actual vs Predicted (Last 30 Days)")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Actual"], mode="lines+markers", name="Actual"))
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Predicted"], mode="lines+markers", name="Predicted"))

    residual_std = backtest_df["Delta"].std()
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Predicted"] + 2 * residual_std, mode="lines", line=dict(width=0)))
    fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["Predicted"] - 2 * residual_std, mode="lines", fill="tonexty", name="±2σ Band", line=dict(width=0)))

    fig_bt.update_layout(
        title="Actual vs Model Predicted (with ±2σ Band)",
        template=PLOTLY_THEME,
        yaxis_title="Price (USD)",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Backtest metrics
    metrics = compute_backtest_metrics(backtest_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{metrics['rmse']:.2f}")
    c2.metric("MAE", f"{metrics['mae']:.2f}")
    c3.metric("MAPE", f"{metrics['mape']:.2f}%")
    c4.metric("Directional Accuracy", f"{dir_acc:.1f}%")

    st.markdown("---")

    # ===================================================================
    # TRADING SIMULATION
    # ===================================================================
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
    fig_eq.add_trace(go.Scatter(x=sim_df.index, y=sim_df["Equity"], mode="lines", name="Equity Curve"))
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


# ===================================================================
# FUTURE FORECAST (10 DAYS)
# ===================================================================
future_df_full = predict_future_prices(ticker, TIMEFRAME_MAPPING[timeframe], days_ahead=10)
last_close_full = get_latest_price(ticker)

if future_df_full is not None and last_close_full is not None:
    st.subheader("🔮 Future Price Forecast (Next 10 Business Days)")

    fig_fut = go.Figure()
    fig_fut.add_trace(go.Scatter(x=future_df_full.index, y=future_df_full["Predicted"], mode="lines+markers", name="Predicted"))
    fig_fut.add_trace(go.Scatter(x=future_df_full.index, y=future_df_full["Upper_CI"], mode="lines", line=dict(width=0)))
    fig_fut.add_trace(go.Scatter(x=future_df_full.index, y=future_df_full["Lower_CI"], mode="lines", fill="tonexty", name="95% CI", line=dict(width=0)))

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
            f"Expected return: {signal_info_full['expected_return_pct']:.2f}% → "
            f"target ≈ ${signal_info_full['target_price']:.2f}"
        ),
    )
else:
    st.info("Not enough data for 10-day forecast.")
