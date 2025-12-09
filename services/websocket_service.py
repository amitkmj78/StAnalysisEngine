# services/websocket_service.py
import json
import threading
import time
from typing import Optional

import streamlit as st

from .config import FINNHUB_API_KEY, WS_PRICE_FEED_URL

try:
    import websocket  # pip install websocket-client
except ImportError:
    websocket = None


def _resolve_ws_url() -> Optional[str]:
    """
    Determine which WebSocket URL to use.
    - If WS_PRICE_FEED_URL is set, use that.
    - Else if FINNHUB_API_KEY is set, use Finnhub trade feed.
    """
    if WS_PRICE_FEED_URL:
        return WS_PRICE_FEED_URL
    if FINNHUB_API_KEY:
        return f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    return None


def is_ws_configured() -> bool:
    """Return True if some WebSocket URL is available."""
    return _resolve_ws_url() is not None and websocket is not None


def start_price_feed(symbol: str):
    """
    Start a background WebSocket listener for the given symbol.
    Updates st.session_state['live_price'] whenever a new tick arrives.
    """

    ws_url = _resolve_ws_url()
    if websocket is None:
        st.warning("websocket-client is not installed. Run `pip install websocket-client`.")
        return
    if not ws_url:
        st.warning("No WebSocket URL configured (FINNHUB_API_KEY or WS_PRICE_FEED_URL missing).")
        return
    if not symbol:
        st.warning("Cannot start live feed: symbol is empty.")
        return

    # Avoid starting multiple threads per symbol
    if "ws_thread" in st.session_state and st.session_state.ws_thread.is_alive():
        return

    st.session_state.setdefault("live_price", None)

    def on_message(ws, message):
        try:
            data = json.loads(message)

            # Finnhub trade format: {"type":"trade","data":[{"p":price, "s":"AAPL", ...}]}
            if isinstance(data, dict) and data.get("type") == "trade":
                trades = data.get("data", [])
                if trades:
                    price = trades[0].get("p")
                    if price is not None:
                        st.session_state.live_price = float(price)
                return

            # Generic provider format: {"symbol":"AAPL","price":195.34}
            if isinstance(data, dict):
                msg_symbol = data.get("symbol", "").upper()
                if msg_symbol == symbol.upper() and "price" in data:
                    st.session_state.live_price = float(data["price"])
        except Exception:
            # swallow parse errors
            pass

    def on_error(ws, err):
        print("WebSocket error:", err)

    def on_close(ws, code, msg):
        print("WebSocket closed:", code, msg)

    def on_open(ws):
        # For Finnhub, subscribe to symbol
        try:
            sub_msg = json.dumps({"type": "subscribe", "symbol": symbol.upper()})
            ws.send(sub_msg)
        except Exception as e:
            print("Subscription error:", e)

    ws_app = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    def _run():
        ws_app.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    st.session_state.ws_thread = thread
    thread.start()
