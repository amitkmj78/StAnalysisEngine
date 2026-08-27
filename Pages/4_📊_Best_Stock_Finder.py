import streamlit as st

from services.stock_finder_service import STOCK_UNIVERSES, rank_stocks, score_stock_ticker
from services.ui_service import apply_app_shell


apply_app_shell(
    "Best Stock Finder",
    "Use the same app-style ranking flow to compare stocks for short-term momentum or long-term investing.",
    accent="#1B7F6B",
)

st.caption("Data source for this page: Yahoo Finance via yfinance.")

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Short term</span>
    <span class="app-pill">Long term</span>
    <span class="app-pill">Momentum</span>
    <span class="app-pill">Quality</span>
</div>
""",
    unsafe_allow_html=True,
)

filter_col1, filter_col2 = st.columns([1, 1])

with filter_col1:
    goal = st.selectbox("Investment horizon", ["Short Term", "Long Term"])

with filter_col2:
    mode = st.selectbox("Lookup mode", ["Rank stock universe", "Search any stock"])

if mode == "Rank stock universe":
    universe = st.selectbox("Stock universe", list(STOCK_UNIVERSES.keys()))
    with st.spinner("Ranking stocks..."):
        ranked = rank_stocks(goal, universe)
    subtitle_context = universe
else:
    stock_ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    with st.spinner("Scoring stock..."):
        ranked = score_stock_ticker(goal, stock_ticker)
    subtitle_context = stock_ticker

if ranked.empty:
    st.warning("No stock data was available for that selection right now. Try another ticker or try again in a moment.")
    st.stop()

winner = ranked.iloc[0]
market_cap_text = f"${winner['Market Cap ($B)']:.1f}B" if winner["Market Cap ($B)"] is not None else "N/A"
rsi_text = f"{winner['RSI']:.2f}" if winner["RSI"] is not None else "N/A"
volatility_text = f"{winner['6M Volatility %']:.2f}%" if winner["6M Volatility %"] is not None else "N/A"
drawdown_text = f"{winner['1Y Max Drawdown %']:.2f}%" if winner["1Y Max Drawdown %"] is not None else "N/A"

st.markdown(
    f"""
<div class="app-card">
    <h3>Top Pick: {winner['Ticker']} - {winner['Name']}</h3>
    <p>This stock ranked highest for the <strong>{goal}</strong> screen inside <strong>{subtitle_context}</strong>.</p>
</div>
""",
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Score", f"{winner['Score']:.1f}/100")
metric_col2.metric("Price", f"${winner['Price']:.2f}")
metric_col3.metric("3M Return", f"{winner['3M Return %']:.2f}%" if winner["3M Return %"] is not None else "N/A")
metric_col4.metric("1Y Return", f"{winner['1Y Return %']:.2f}%" if winner["1Y Return %"] is not None else "N/A")

detail_col1, detail_col2 = st.columns([1.15, 0.85])

with detail_col1:
    st.markdown(
        f"""
<div class="app-card">
    <h3>Why It Ranked First</h3>
    <p><strong>Sector:</strong> {winner['Sector']}</p>
    <p><strong>Market cap:</strong> {market_cap_text}</p>
    <p><strong>RSI:</strong> {rsi_text}</p>
    <p><strong>6M volatility:</strong> {volatility_text}</p>
    <p><strong>1Y max drawdown:</strong> {drawdown_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )

with detail_col2:
    st.markdown(
        """
<div class="app-card">
    <h3>How The Ranking Works</h3>
    <p><strong>Short Term</strong> favors recent momentum, RSI balance, MACD strength, and volume.</p>
    <p><strong>Long Term</strong> puts more weight on 1Y and 3Y performance, growth, valuation, and drawdown.</p>
</div>
""",
        unsafe_allow_html=True,
    )

display_df = ranked[
    [
        "Ticker",
        "Name",
        "Sector",
        "Score",
        "Price",
        "1M Return %",
        "3M Return %",
        "6M Return %",
        "1Y Return %",
        "3Y Annualized %",
        "RSI",
        "Volume Strength %",
        "6M Volatility %",
        "1Y Max Drawdown %",
        "Revenue Growth %",
        "Earnings Growth %",
        "Forward PE",
        "Market Cap ($B)",
    ]
].copy()

st.markdown(
    """
<div class="app-card">
    <h3>Comparison Table</h3>
    <p>Use this shortlist to compare momentum, risk, growth, and valuation side by side.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(format="%.1f"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "1M Return %": st.column_config.NumberColumn(format="%.2f%%"),
        "3M Return %": st.column_config.NumberColumn(format="%.2f%%"),
        "6M Return %": st.column_config.NumberColumn(format="%.2f%%"),
        "1Y Return %": st.column_config.NumberColumn(format="%.2f%%"),
        "3Y Annualized %": st.column_config.NumberColumn(format="%.2f%%"),
        "RSI": st.column_config.NumberColumn(format="%.2f"),
        "Volume Strength %": st.column_config.NumberColumn(format="%.2f%%"),
        "6M Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
        "1Y Max Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
        "Revenue Growth %": st.column_config.NumberColumn(format="%.2f%%"),
        "Earnings Growth %": st.column_config.NumberColumn(format="%.2f%%"),
        "Forward PE": st.column_config.NumberColumn(format="%.2f"),
        "Market Cap ($B)": st.column_config.NumberColumn(format="$%.1fB"),
    },
)

st.markdown(
    """
<div class="app-card">
    <h3>Final Note</h3>
    <p>This is a ranking tool, not a guarantee. Use it to narrow the list, then check earnings, news, and your risk tolerance before making a trade or investment decision.</p>
</div>
""",
    unsafe_allow_html=True,
)
