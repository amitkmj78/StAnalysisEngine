import plotly.graph_objects as go
from plotly.colors import qualitative
import streamlit as st

from services.million_plan_service import (
    DEFAULT_TARGET_AMOUNT,
    DEFAULT_TARGET_YEARS,
    build_million_plan_table_from_returns,
    get_million_plan_picks,
    project_total_future_value,
    required_monthly_investment,
)
from services.stock_finder_service import STOCK_UNIVERSES
from services.ui_service import apply_app_shell

FUND_GOALS = ["Balanced Core", "Lowest Cost", "Best Growth", "Most Stable"]
FUND_CATEGORIES = ["All", "US Large Blend", "US Total Market", "US Growth", "US Small Cap", "International", "Bond"]
STOCK_GOALS = ["Short Term", "Long Term"]

apply_app_shell(
    "Strategies",
    "Build a target-based investing strategy with dynamic time horizon and see the best fund and stock candidates that can help support the plan.",
    accent="#C76B17",
)

st.caption("Data source for ranked picks on this page: Yahoo Finance via yfinance.")

st.markdown(
    """
<div class="app-pill-row">
    <span class="app-pill">Dynamic target</span>
    <span class="app-pill">Dynamic years</span>
    <span class="app-pill">Contribution math</span>
    <span class="app-pill">Best fund + stock</span>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    target_amount = st.number_input("Target amount", min_value=50000, max_value=10000000, value=DEFAULT_TARGET_AMOUNT, step=50000)
with col2:
    years = st.slider("Years to goal", min_value=1, max_value=20, value=DEFAULT_TARGET_YEARS)
with col3:
    starting_capital = st.number_input("Starting capital", min_value=0, max_value=500000, value=0, step=1000)

col4, col5 = st.columns([1, 1])
with col4:
    custom_return = st.slider("Custom annual return assumption", min_value=4, max_value=20, value=10)
with col5:
    top_n = st.slider("Number of live picks", min_value=1, max_value=5, value=2)

scenario_col1, scenario_col2, scenario_col3 = st.columns([1, 1, 1])
with scenario_col1:
    min_return = st.slider("Min return case", min_value=4, max_value=18, value=6)
with scenario_col2:
    max_return = st.slider("Max return case", min_value=min_return + 1, max_value=20, value=15)
with scenario_col3:
    return_step = st.selectbox("Return step", [1, 2, 3], index=1)

fund_col1, fund_col2 = st.columns([1, 1])
with fund_col1:
    fund_goal = st.selectbox("Fund strategy source", FUND_GOALS)
with fund_col2:
    fund_category = st.selectbox("Fund category source", FUND_CATEGORIES)

stock_col1, stock_col2, stock_col3 = st.columns([1, 1, 1])
with stock_col1:
    stock_goal = st.selectbox("Stock strategy source", STOCK_GOALS)
with stock_col2:
    stock_universe = st.selectbox("Stock universe source", list(STOCK_UNIVERSES.keys()))
with stock_col3:
    preferred_builder = st.selectbox("Primary builder", ["Best Fund", "Best Stock", "Show Both"])

return_cases = list(range(int(min_return), int(max_return) + 1, int(return_step)))
if float(custom_return) not in return_cases:
    return_cases.append(float(custom_return))

plan_df = build_million_plan_table_from_returns(
    annual_returns=return_cases,
    target_amount=float(target_amount),
    years=int(years),
    starting_capital=float(starting_capital),
)
custom_monthly = required_monthly_investment(
    target_amount=float(target_amount),
    years=int(years),
    annual_return_pct=float(custom_return),
    starting_capital=float(starting_capital),
)

st.markdown(
    f"""
<div class="app-card">
    <h3>What It Takes</h3>
    <p>To target <strong>${target_amount:,.0f}</strong> in <strong>{years} years</strong>, your monthly contribution requirement depends heavily on return assumptions and starting capital.</p>
</div>
""",
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Target", f"${target_amount:,.0f}")
metric_col2.metric("Time Horizon", f"{years} years")
metric_col3.metric("Needed at {0}%".format(custom_return), f"${custom_monthly:,.0f}/mo")

st.markdown(
    """
<div class="app-card">
    <h3>Required Monthly Contribution By Strategy</h3>
    <p>These scenarios are simple planning cases, not guarantees. Higher expected return assumptions lower the monthly contribution required, but usually come with higher risk and drawdown potential.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.dataframe(
    plan_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Annual Return %": st.column_config.NumberColumn(format="%.1f%%"),
        "Required Monthly Invest": st.column_config.NumberColumn(format="$%.0f"),
        "Total Contributions": st.column_config.NumberColumn(format="$%.0f"),
        "Projected Value": st.column_config.NumberColumn(format="$%.0f"),
    },
)

fig = go.Figure()
palette = qualitative.Set2 + qualitative.Safe + qualitative.Bold
bar_colors = [palette[i % len(palette)] for i in range(len(plan_df))]
fig.add_trace(
    go.Bar(
        x=plan_df["Strategy"],
        y=plan_df["Required Monthly Invest"],
        marker_color=bar_colors,
        name="Required Monthly Invest",
    )
)
fig.update_layout(
    title=f"Monthly Contribution Needed to Reach ${target_amount:,.0f} in {years} Years",
    yaxis_title="USD per month",
    template="plotly_white",
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

picks = get_million_plan_picks(
    fund_goal=fund_goal,
    fund_category=fund_category,
    stock_goal=stock_goal,
    stock_universe=stock_universe,
    top_n=top_n,
)

st.markdown(
    """
<div class="app-card">
    <h3>Best Builders Right Now</h3>
    <p>These cards are built live from the current top-ranked fund and stock results using your selected screens above.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not picks:
    st.warning("No ranked picks were available right now.")
else:
    fund_picks = [pick for pick in picks if pick.asset_type == "Fund"]
    stock_picks = [pick for pick in picks if pick.asset_type == "Stock"]

    selected_picks = picks
    if preferred_builder == "Best Fund":
        selected_picks = fund_picks[:top_n]
    elif preferred_builder == "Best Stock":
        selected_picks = stock_picks[:top_n]

    if preferred_builder in {"Best Fund", "Show Both"} and fund_picks:
        best_fund = fund_picks[0]
        best_fund_monthly = (
            required_monthly_investment(
                target_amount=float(target_amount),
                years=int(years),
                annual_return_pct=best_fund.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if best_fund.annual_return_pct is not None
            else None
        )
        best_fund_future_value = (
            project_total_future_value(
                monthly_amount=best_fund_monthly,
                years=int(years),
                annual_return_pct=best_fund.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if best_fund_monthly is not None
            else None
        )
        st.markdown(
            f"""
<div class="app-card">
    <h3>Best Fund To Help Build: {best_fund.ticker} - {best_fund.name}</h3>
    <p><strong>Ranking score:</strong> {best_fund.score:.1f}/100</p>
    <p><strong>Historic annualized return used:</strong> {f"{best_fund.annual_return_pct:.2f}%" if best_fund.annual_return_pct is not None else "N/A"}</p>
    <p><strong>Estimated monthly amount needed:</strong> {f"${best_fund_monthly:,.0f}/mo" if best_fund_monthly is not None else "N/A"}</p>
    <p><strong>Future value built from historic return:</strong> {f"${best_fund_future_value:,.0f}" if best_fund_future_value is not None else "N/A"}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    if preferred_builder in {"Best Stock", "Show Both"} and stock_picks:
        best_stock = stock_picks[0]
        best_stock_monthly = (
            required_monthly_investment(
                target_amount=float(target_amount),
                years=int(years),
                annual_return_pct=best_stock.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if best_stock.annual_return_pct is not None
            else None
        )
        best_stock_future_value = (
            project_total_future_value(
                monthly_amount=best_stock_monthly,
                years=int(years),
                annual_return_pct=best_stock.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if best_stock_monthly is not None
            else None
        )
        st.markdown(
            f"""
<div class="app-card">
    <h3>Best Stock To Help Build: {best_stock.ticker} - {best_stock.name}</h3>
    <p><strong>Ranking score:</strong> {best_stock.score:.1f}/100</p>
    <p><strong>Historic annualized return used:</strong> {f"{best_stock.annual_return_pct:.2f}%" if best_stock.annual_return_pct is not None else "N/A"}</p>
    <p><strong>Estimated monthly amount needed:</strong> {f"${best_stock_monthly:,.0f}/mo" if best_stock_monthly is not None else "N/A"}</p>
    <p><strong>Future value built from historic return:</strong> {f"${best_stock_future_value:,.0f}" if best_stock_future_value is not None else "N/A"}</p>
</div>
""",
            unsafe_allow_html=True,
        )

    for pick in selected_picks:
        implied_monthly = (
            required_monthly_investment(
                target_amount=float(target_amount),
                years=int(years),
                annual_return_pct=pick.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if pick.annual_return_pct is not None
            else None
        )
        projected_value = (
            project_total_future_value(
                monthly_amount=implied_monthly,
                years=int(years),
                annual_return_pct=pick.annual_return_pct,
                starting_capital=float(starting_capital),
            )
            if implied_monthly is not None
            else None
        )
        st.markdown(
            f"""
<div class="app-card">
    <h3>{pick.label}: {pick.ticker} - {pick.name}</h3>
    <p><strong>Type:</strong> {pick.asset_type}</p>
    <p><strong>Ranking score:</strong> {pick.score:.1f}/100</p>
    <p><strong>Historic annualized return used:</strong> {f"{pick.annual_return_pct:.2f}%" if pick.annual_return_pct is not None else "N/A"}</p>
    <p><strong>Monthly amount needed to aim for the target:</strong> {f"${implied_monthly:,.0f}/mo" if implied_monthly is not None else "N/A"}</p>
    <p><strong>Projected future value from historic return:</strong> {f"${projected_value:,.0f}" if projected_value is not None else "N/A"}</p>
</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    """
<div class="app-card">
    <h3>Practical Ways to Improve the Odds</h3>
    <p><strong>Raise starting capital:</strong> A larger upfront amount lowers the monthly burden immediately.</p>
    <p><strong>Increase monthly deposits over time:</strong> Even small annual step-ups can materially change the result.</p>
    <p><strong>Favor consistency over perfect timing:</strong> Missing deposits usually hurts more than trying to optimize each entry.</p>
    <p><strong>Match risk to reality:</strong> A 5-year $1M goal is demanding, so aggressive assumptions should be treated carefully.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="app-card">
    <h3>Final Note</h3>
    <p>This page is a planning calculator, not a promise. It uses simplified compounding math and current ranking outputs, and it does not account for taxes, slippage, dividends, or changing market regimes.</p>
</div>
""",
    unsafe_allow_html=True,
)
