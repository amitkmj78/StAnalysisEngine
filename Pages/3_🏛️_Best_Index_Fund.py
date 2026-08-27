import streamlit as st

from services.index_fund_service import rank_index_funds, score_fund_ticker
from services.ui_service import apply_app_shell


apply_app_shell(
    "Best Index Fund Finder",
    "Compare major index ETFs in a cleaner app-style view and rank them by the goal that matters most to you.",
)

st.caption("Data source for this page: Yahoo Finance via yfinance.")

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Balanced core</span>
    <span class="app-pill">Lowest cost</span>
    <span class="app-pill">Best growth</span>
    <span class="app-pill">Most stable</span>
</div>
""",
    unsafe_allow_html=True,
)

filter_col1, filter_col2 = st.columns([1, 1])

with filter_col1:
    goal = st.selectbox(
        "What are you optimizing for?",
        ["Balanced Core", "Lowest Cost", "Best Growth", "Most Stable"],
    )

with filter_col2:
    mode = st.selectbox("Lookup mode", ["Rank curated funds", "Search any fund"])

if mode == "Rank curated funds":
    category = st.selectbox(
        "Fund category",
        ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"],
    )
    with st.spinner("Ranking index funds..."):
        ranked = rank_index_funds(goal, category)
else:
    fund_ticker = st.text_input("Fund ticker", value="VOO").strip().upper()
    with st.spinner("Scoring fund..."):
        ranked = score_fund_ticker(goal, fund_ticker)

if ranked.empty:
    st.warning("No fund data was available for that selection right now. Try another ticker or try again in a moment.")
    st.stop()

winner = ranked.iloc[0]
volatility_text = f"{winner['1Y Volatility %']:.2f}%" if winner["1Y Volatility %"] is not None else "N/A"
drawdown_text = f"{winner['3Y Max Drawdown %']:.2f}%" if winner["3Y Max Drawdown %"] is not None else "N/A"
assets_text = f"${winner['Assets ($B)']:.1f}B" if winner["Assets ($B)"] is not None else "N/A"

st.markdown(
    f"""
<div class="app-card">
    <h3>Top Pick: {winner['Ticker']} - {winner['Fund']}</h3>
    <p>This fund scored highest for <strong>{goal}</strong> in the current result set.</p>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Score", f"{winner['Score']:.1f}/100")
col2.metric("Expense Ratio", f"{winner['Expense Ratio %']:.3f}%" if winner["Expense Ratio %"] is not None else "N/A")
col3.metric("1Y Return", f"{winner['1Y Return %']:.2f}%")
col4.metric("3Y Annualized", f"{winner['3Y Annualized %']:.2f}%" if winner["3Y Annualized %"] is not None else "N/A")

detail_col1, detail_col2 = st.columns([1.2, 0.8])

with detail_col1:
    st.markdown(
        f"""
<div class="app-card">
    <h3>Why It Landed First</h3>
    <p><strong>Benchmark:</strong> {winner['Benchmark']}</p>
    <p><strong>Category:</strong> {winner['Category']}</p>
    <p><strong>1Y volatility:</strong> {volatility_text}</p>
    <p><strong>3Y max drawdown:</strong> {drawdown_text}</p>
    <p><strong>Assets:</strong> {assets_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )

with detail_col2:
    st.markdown(
        """
<div class="app-card">
    <h3>How To Use This</h3>
    <p>Start with the top few names, then pick based on fund access, tax location, and whether you want broad market coverage or a specific slice.</p>
</div>
""",
        unsafe_allow_html=True,
    )

display_df = ranked[
    [
        "Ticker",
        "Fund",
        "Benchmark",
        "Category",
        "Score",
        "Expense Ratio %",
        "1Y Return %",
        "3Y Annualized %",
        "1Y Volatility %",
        "3Y Max Drawdown %",
        "Assets ($B)",
    ]
].copy()

st.markdown(
    """
<div class="app-card">
    <h3>Comparison Table</h3>
    <p>Use this shortlist to compare cost, return, volatility, drawdown, and size side by side.</p>
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
        "Expense Ratio %": st.column_config.NumberColumn(format="%.3f%%"),
        "1Y Return %": st.column_config.NumberColumn(format="%.2f%%"),
        "3Y Annualized %": st.column_config.NumberColumn(format="%.2f%%"),
        "1Y Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
        "3Y Max Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
        "Assets ($B)": st.column_config.NumberColumn(format="$%.1fB"),
    },
)

st.markdown(
    """
<div class="app-card">
    <h3>Final Note</h3>
    <p>Use this as a shortlist tool. Funds tracking the same index can be nearly interchangeable, so expense ratio, liquidity, and account availability usually decide the final pick.</p>
</div>
""",
    unsafe_allow_html=True,
)
