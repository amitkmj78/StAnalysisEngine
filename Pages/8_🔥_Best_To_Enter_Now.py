import plotly.graph_objects as go
import streamlit as st

from services.entry_strategy_service import (
    ENTRY_FUND_UNIVERSES,
    ENTRY_STOCK_UNIVERSES,
    build_entry_plan,
    scan_best_entries,
)
from services.ui_service import apply_app_shell


apply_app_shell(
    "Best To Enter Now",
    "Scan current fund or stock setups and surface the strongest entry candidates right now using trend, support, resistance, and momentum from Yahoo Finance data.",
    accent="#B23A48",
)

st.caption("Data source for this page: Yahoo Finance via yfinance.")

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Best current entries</span>
    <span class="app-pill">Funds or stocks</span>
    <span class="app-pill">Momentum + support</span>
    <span class="app-pill">Entry score</span>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    asset_type = st.selectbox("Asset type", ["Fund", "Stock"])
with col2:
    mode = st.selectbox("Mode", ["Scan current best entries", "Check one ticker"])
with col3:
    top_n = st.slider("How many results", min_value=3, max_value=15, value=5)

if mode == "Scan current best entries":
    if asset_type == "Fund":
        universe = st.selectbox("Fund universe", list(ENTRY_FUND_UNIVERSES.keys()))
    else:
        universe = st.selectbox("Stock universe", list(ENTRY_STOCK_UNIVERSES.keys()))

    with st.spinner("Scanning current setups..."):
        ranked = scan_best_entries(asset_type, universe)

    if ranked.empty:
        st.warning("No entry setups were available right now.")
        st.stop()

    ranked = ranked.head(top_n)
    winner = ranked.iloc[0]

    st.markdown(
        f"""
<div class="app-card">
    <h3>Top Entry Right Now: {winner['Ticker']}</h3>
    <p><strong>Signal:</strong> {winner['Signal']}</p>
    <p>This is the strongest current <strong>{asset_type.lower()}</strong> entry setup in <strong>{universe}</strong> based on the current scan.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Entry Score", f"{winner['Entry Score']:.1f}/100")
    metric_col2.metric("Current Price", f"${winner['Current Price']:.2f}")
    metric_col3.metric("Entry Zone", f"${winner['Entry Low']:.2f} - ${winner['Entry High']:.2f}")
    metric_col4.metric("Breakout", f"${winner['Breakout Entry']:.2f}")

    st.markdown(
        """
<div class="app-card">
    <h3>Best Current Entries</h3>
    <p>Higher scores mean a cleaner mix of trend, momentum, and location near support or a valid breakout level.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.dataframe(
        ranked,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry Score": st.column_config.NumberColumn(format="%.1f"),
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Entry Low": st.column_config.NumberColumn(format="$%.2f"),
            "Entry High": st.column_config.NumberColumn(format="$%.2f"),
            "Breakout Entry": st.column_config.NumberColumn(format="$%.2f"),
            "Stop Loss": st.column_config.NumberColumn(format="$%.2f"),
            "First Target": st.column_config.NumberColumn(format="$%.2f"),
            "RSI": st.column_config.NumberColumn(format="%.2f"),
            "Support 20D": st.column_config.NumberColumn(format="$%.2f"),
            "Resistance 20D": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
else:
    ticker = st.text_input("Ticker", value="AAPL" if asset_type == "Stock" else "VOO").strip().upper()
    with st.spinner("Checking entry timing..."):
        plan = build_entry_plan(ticker)

    if plan is None:
        st.warning("Not enough price history was available for that ticker. Try another one.")
        st.stop()

    st.markdown(
        f"""
<div class="app-card">
    <h3>{plan['ticker']} Entry Snapshot</h3>
    <p><strong>Signal:</strong> {plan['signal']}</p>
    <p>{plan['summary']}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Entry Score", f"{plan['entry_score']:.1f}/100")
    metric_col2.metric("Current Price", f"${plan['current_price']:.2f}")
    metric_col3.metric("Entry Zone", f"${plan['ideal_entry_low']:.2f} - ${plan['ideal_entry_high']:.2f}")
    metric_col4.metric("Breakout", f"${plan['breakout_entry']:.2f}")

    detail_col1, detail_col2 = st.columns([1.2, 0.8])
    with detail_col1:
        st.markdown(
            f"""
<div class="app-card">
    <h3>Entry Levels</h3>
    <p><strong>Buy zone:</strong> ${plan['ideal_entry_low']:.2f} - ${plan['ideal_entry_high']:.2f}</p>
    <p><strong>Breakout trigger:</strong> ${plan['breakout_entry']:.2f}</p>
    <p><strong>Stop loss:</strong> ${plan['stop_loss']:.2f}</p>
    <p><strong>First target:</strong> ${plan['first_target']:.2f}</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with detail_col2:
        st.markdown(
            f"""
<div class="app-card">
    <h3>Trend Read</h3>
    <p><strong>RSI:</strong> {f"{plan['rsi']:.2f}" if plan['rsi'] is not None else "N/A"}</p>
    <p><strong>Short-term trend:</strong> {"Uptrend" if plan['trend_up'] else "Mixed / weak trend"}</p>
    <p><strong>Long-term trend:</strong> {"Long-term uptrend" if plan['long_term_up'] else "Long-term trend not fully supportive"}</p>
    <p><strong>20D support:</strong> ${plan['support_20']:.2f}</p>
    <p><strong>20D resistance:</strong> ${plan['resistance_20']:.2f}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    history = plan["history"].tail(120).copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#1c3556", width=2.5),
        )
    )

    for label, value, color in [
        ("Entry Low", plan["ideal_entry_low"], "#138A72"),
        ("Entry High", plan["ideal_entry_high"], "#2FAE8F"),
        ("Breakout", plan["breakout_entry"], "#C76B17"),
        ("Stop", plan["stop_loss"], "#B23A48"),
        ("Target", plan["first_target"], "#6B8E23"),
    ]:
        fig.add_hline(y=value, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")

    fig.update_layout(
        title=f"{plan['ticker']} Price and Entry Levels",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    volume_strength = (
        (plan["latest_volume"] / plan["avg_volume_20"] - 1) * 100
        if plan["latest_volume"] is not None and plan["avg_volume_20"] not in (None, 0)
        else None
    )
    volume_text = f"{volume_strength:.1f}%" if volume_strength is not None else "N/A"

    st.markdown(
        f"""
<div class="app-card">
    <h3>What To Watch Before Entering</h3>
    <p><strong>Momentum:</strong> MACD {f"{plan['macd']:.2f}" if plan['macd'] is not None else "N/A"} vs signal {f"{plan['macd_signal']:.2f}" if plan['macd_signal'] is not None else "N/A"}</p>
    <p><strong>Volume vs 20-day average:</strong> {volume_text}</p>
    <p><strong>ATR volatility:</strong> {f"${plan['atr']:.2f}" if plan['atr'] is not None else "N/A"}</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="app-card">
    <h3>How To Use The Signals</h3>
    <p><strong>Buy Now:</strong> trend and momentum are aligned and price is not too extended.</p>
    <p><strong>Buy on Pullback:</strong> wait for a dip into support.</p>
    <p><strong>Breakout Entry:</strong> wait for price to clear resistance.</p>
    <p><strong>Wait:</strong> the setup is still messy.</p>
</div>
""",
    unsafe_allow_html=True,
)
