# services/your_chart_helpers.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.data_service import get_stock_data, get_latest_price
from services.prediction_service import predict_next_price


# ============================================================
#  PRICE CHART RENDERER (original app logic preserved)
# ============================================================
def render_price_chart(ticker: str, timeframe_label: str, theme: str = None):
    """
    Renders a candlestick chart + last close + predicted next-day price.
    Returns (latest_price, predicted_price)
    """

    if theme is None:
        theme = "plotly_dark" if st.session_state.get("theme") == "Dark Mode" else "plotly_white"

    # Load OHLC data
    data = get_stock_data(ticker, timeframe_label)
    if data.empty:
        st.warning(f"No data available for {ticker} for timeframe {timeframe_label}.")
        return None, None

    latest_price = get_latest_price(ticker)
    predicted_price = predict_next_price(ticker, timeframe_label)

    # Plot
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
        template=theme,
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    return latest_price, predicted_price


# ============================================================
#  SIMPLE LONG-ONLY TRADING SIMULATION
# ============================================================
def run_trading_sim(backtest_df: pd.DataFrame, threshold_pct: float = 0.0):
    """
    Very simple strategy:

    - Buy at close(t) if predicted(t+1) >= actual(t) * (1 + threshold%)
    - Exit next day at close(t+1)
    
    Returns:
        sim_df: dataframe with returns + equity curve
        stats: dict with win rate, total return, trade count
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
