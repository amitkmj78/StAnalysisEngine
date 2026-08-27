import plotly.graph_objects as go
import streamlit as st

from services.monthly_investing_service import (
    get_best_monthly_pick,
    project_future_value,
    simulate_monthly_plan,
)
from services.stock_finder_service import STOCK_UNIVERSES
from services.ui_service import apply_app_shell


FUND_GOALS = ["Balanced Core", "Lowest Cost", "Best Growth", "Most Stable"]
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]
STOCK_GOALS = ["Short Term", "Long Term"]


apply_app_shell(
    "Monthly Investing Plan",
    "See what investing $1,000 every month into a top-ranked fund or stock could look like over time.",
    accent="#138A72",
)

st.caption("Data source for this page: Yahoo Finance via yfinance.")

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">$1,000/month</span>
    <span class="app-pill">Best fund</span>
    <span class="app-pill">Best stock</span>
    <span class="app-pill">DCA plan</span>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    asset_type = st.selectbox("Asset type", ["Fund", "Stock"])

with col2:
    monthly_amount = st.number_input("Monthly amount", min_value=100, max_value=10000, value=1000, step=100)

with col3:
    years = st.slider("Plan length", min_value=1, max_value=15, value=5)

if asset_type == "Fund":
    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        goal = st.selectbox("Fund goal", FUND_GOALS)
    with filter_col2:
        selection = st.selectbox("Fund category", FUND_CATEGORIES)
else:
    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        goal = st.selectbox("Stock goal", STOCK_GOALS)
    with filter_col2:
        selection = st.selectbox("Stock universe", list(STOCK_UNIVERSES.keys()))

with st.spinner("Building your plan..."):
    recommendation = get_best_monthly_pick(asset_type, goal, selection)

if recommendation is None:
    st.warning("No ranked pick was available right now. Try a different filter or try again in a moment.")
    st.stop()

history_df, summary = simulate_monthly_plan(recommendation.ticker, float(monthly_amount), years)
projected_value = project_future_value(float(monthly_amount), years, recommendation.expected_return_pct)

st.markdown(
    f"""
<div class="app-card">
    <h3>Suggested Pick: {recommendation.ticker} - {recommendation.name}</h3>
    <p>This {recommendation.asset_type.lower()} ranked highest for <strong>{goal}</strong> in the current screen and is being used for the monthly investing plan.</p>
</div>
""",
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Score", f"{recommendation.score:.1f}/100")
metric_col2.metric("Monthly Invest", f"${monthly_amount:,.0f}")
metric_col3.metric("Plan Length", f"{years} years")
metric_col4.metric(
    "Expected Annual Return",
    f"{recommendation.expected_return_pct:.2f}%" if recommendation.expected_return_pct is not None else "N/A",
)

if not summary:
    st.warning("Not enough price history was available to simulate the monthly plan.")
    st.stop()

metric_col5, metric_col6, metric_col7 = st.columns(3)
metric_col5.metric("Total Invested", f"${summary['total_invested']:,.0f}")
metric_col6.metric("Historical Ending Value", f"${summary['ending_value']:,.0f}")
metric_col7.metric("Historical Gain", f"${summary['gain']:,.0f} ({summary['gain_pct']:.1f}%)")

detail_col1, detail_col2 = st.columns([1.2, 0.8])

with detail_col1:
    st.markdown(
        f"""
<div class="app-card">
    <h3>How To Read This</h3>
    <p><strong>Historical ending value</strong> shows what regular monthly investing would have grown to over the last {summary['months']} months using actual price history for {recommendation.ticker}.</p>
    <p><strong>Projected value</strong> uses the asset's trailing annualized return as a simple forward estimate, which is useful for planning but not guaranteed.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with detail_col2:
    st.markdown(
        f"""
<div class="app-card">
    <h3>Forward Estimate</h3>
    <p><strong>Projected portfolio value:</strong> {f"${projected_value:,.0f}" if projected_value is not None else "N/A"}</p>
    <p><strong>Current price used:</strong> ${summary['latest_price']:.2f}</p>
</div>
""",
        unsafe_allow_html=True,
    )

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_df["Date"],
        y=history_df["Portfolio Value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#138A72", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=history_df["Date"],
        y=history_df["Total Invested"],
        mode="lines",
        name="Total Invested",
        line=dict(color="#1c3556", width=2, dash="dash"),
    )
)
fig.update_layout(
    title=f"{recommendation.ticker} Monthly Investing Path",
    xaxis_title="Date",
    yaxis_title="USD",
    template="plotly_white",
    height=460,
)
st.plotly_chart(fig, use_container_width=True)

display_df = history_df[["Date", "Monthly Contribution", "Price", "Shares Bought", "Total Invested", "Portfolio Value"]].copy()
display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")

st.markdown(
    """
<div class="app-card">
    <h3>Monthly Contribution History</h3>
    <p>Review the month-by-month contribution path and how portfolio value compared with total capital invested.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Monthly Contribution": st.column_config.NumberColumn(format="$%.0f"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Shares Bought": st.column_config.NumberColumn(format="%.4f"),
        "Total Invested": st.column_config.NumberColumn(format="$%.0f"),
        "Portfolio Value": st.column_config.NumberColumn(format="$%.0f"),
    },
)

st.markdown(
    """
<div class="app-card">
    <h3>Final Note</h3>
    <p>This planner is a simple dollar-cost averaging view. It does not account for taxes, fees, dividends, or future regime changes, so use it as a planning tool rather than a promise.</p>
</div>
""",
    unsafe_allow_html=True,
)
