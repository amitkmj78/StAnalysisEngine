import streamlit as st
from services.data_service import get_latest_price
from services.prediction_service import predict_next_price
from services.analysis_service import get_analysis
from Agent.meta_agent import get_debug_logs

st.title("🏠 Overview")

ticker = st.session_state.get("ticker")
timeframe = st.session_state.get("timeframe")

st.subheader(f"Overview — {ticker}")

latest = get_latest_price(ticker)
pred = predict_next_price(ticker, timeframe)

col1, col2 = st.columns(2)
col1.metric("Last Close", f"${latest:.2f}" if latest else "N/A")
col2.metric("Predicted Next Price", f"${pred:.2f}" if pred else "N/A")

st.markdown("### Quick Company Snapshot")
snapshot = get_analysis("Basic Info", ticker, None)
st.write(snapshot)

st.markdown("### Debug Logs")
st.text_area("Meta-Agent Logs", get_debug_logs(), height=300)
