import plotly.graph_objects as go
import streamlit as st

from services.data_service import TIMEFRAME_MAPPING, get_latest_price
from services.prediction_service import (
    compute_backtest_metrics,
    generate_trading_signal,
    predict_backtest_prices,
    predict_future_prices,
    predict_next_price,
)
from services.ui_service import apply_app_shell


apply_app_shell(
    "AI Price Prediction",
    "A backtested quant forecast for one ticker, shown next to how often it has actually beaten doing nothing.",
    accent="#2F4A73",
)

st.caption(
    "Data source for this page: Yahoo Finance via yfinance. Model: gradient-boosted "
    "regressor on RSI/MACD/Bollinger position and lagged returns, retrained periodically per ticker."
)

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Next-day forecast</span>
    <span class="app-pill">10-day outlook</span>
    <span class="app-pill">Backtested accuracy</span>
    <span class="app-pill">Naive baseline disclosed</span>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
with col2:
    timeframe_options = list(TIMEFRAME_MAPPING.keys())
    default_index = timeframe_options.index("1 Year") if "1 Year" in timeframe_options else 0
    timeframe = st.selectbox("Historical window used to train the model", timeframe_options, index=default_index)

if not ticker:
    st.info("Enter a ticker above to get a prediction.")
    st.stop()

period = TIMEFRAME_MAPPING[timeframe]

with st.spinner(f"Training model and forecasting {ticker}..."):
    last_close = get_latest_price(ticker)
    next_price = predict_next_price(ticker, period)
    future_df = predict_future_prices(ticker, period, days_ahead=10)
    backtest_df = predict_backtest_prices(ticker, period, days_back=30)

if last_close is None:
    st.warning("No price data was available for that ticker. Try another one.")
    st.stop()

next_price_html = f" &middot; Predicted next close: <strong>${next_price:.2f}</strong>" if next_price is not None else ""
st.markdown(
    f"""
<div class="app-card">
    <h3>{ticker} — Next-Day Forecast</h3>
    <p>Last close: <strong>${last_close:.2f}</strong>{next_price_html}</p>
</div>
""",
    unsafe_allow_html=True,
)

signal_info = generate_trading_signal(last_close, future_df) if future_df is not None else None
if signal_info is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("10-Day Signal", signal_info["signal"])
    m2.metric("Expected Return (10d)", f"{signal_info['expected_return_pct']:.2f}%")
    m3.metric("Target Price", f"${signal_info['target_price']:.2f}")
    st.caption(
        "This is one data-driven signal, not a guarantee — see 'Backtest Accuracy' "
        "below for how it has historically performed against a simple no-change baseline."
    )

if future_df is not None:
    fig_fut = go.Figure()
    fig_fut.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#2F4A73", width=2.5),
        )
    )
    fig_fut.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Upper_CI"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig_fut.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Lower_CI"],
            mode="lines",
            fill="tonexty",
            name="95% CI",
            line=dict(width=0),
            fillcolor="rgba(47, 74, 115, 0.15)",
        )
    )
    fig_fut.update_layout(
        title=f"{ticker} — 10-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_fut, use_container_width=True)
else:
    st.info("Not enough price history for a 10-day forecast.")

st.markdown("---")
st.subheader("🎯 Backtest Accuracy")

if backtest_df is None or backtest_df.empty:
    st.info("Not enough price history for a walk-forward backtest.")
else:
    fig_bt = go.Figure()
    fig_bt.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#14213D"),
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#2F4A73"),
        )
    )
    fig_bt.add_trace(
        go.Scatter(
            x=backtest_df.index,
            y=backtest_df["Naive"],
            mode="lines",
            name="Naive (no-change)",
            line=dict(dash="dot", color="#9AA5B1"),
        )
    )
    fig_bt.update_layout(
        title=f"{ticker} — Last 30 Days: Actual vs Predicted vs Naive",
        template="plotly_white",
        yaxis_title="Price (USD)",
        height=420,
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    metrics = compute_backtest_metrics(backtest_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['rmse']:.2f}")
    c2.metric("MAE", f"{metrics['mae']:.2f}")
    c3.metric("MAPE", f"{metrics['mape']:.2f}%")

    if "naive_rmse" in metrics:
        st.caption(
            "Naive baseline = predicting no price change (tomorrow's price = today's "
            "close). Delta shown as model minus naive, so negative is better."
        )
        n1, n2, n3, n4 = st.columns(4)
        n1.metric(
            "RMSE vs Naive",
            f"{metrics['naive_rmse']:.2f}",
            delta=f"{metrics['rmse'] - metrics['naive_rmse']:.2f}",
            delta_color="inverse",
        )
        n2.metric(
            "MAE vs Naive",
            f"{metrics['naive_mae']:.2f}",
            delta=f"{metrics['mae'] - metrics['naive_mae']:.2f}",
            delta_color="inverse",
        )
        n3.metric(
            "MAPE vs Naive",
            f"{metrics['naive_mape']:.2f}%",
            delta=f"{metrics['mape'] - metrics['naive_mape']:.2f}%",
            delta_color="inverse",
        )
        if metrics["beats_naive"]:
            n4.success("Beats naive baseline")
        else:
            n4.error("Does not beat naive")

st.markdown(
    """
<div class="app-card">
    <h3>How To Read This Page</h3>
    <p>The forecast comes from a gradient-boosted model trained on this ticker's own recent RSI, MACD, Bollinger position, and lagged returns — not on news, filings, or sentiment.</p>
    <p>The naive baseline is simply "assume tomorrow's price equals today's close." Across most tickers this model does not reliably beat that baseline — the metrics above are the actual, current test for this ticker, not a marketing claim.</p>
</div>
""",
    unsafe_allow_html=True,
)
