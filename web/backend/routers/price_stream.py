"""PLT-3: live price updates pushed over a WebSocket instead of each client
polling GET /api/v1/search/price on its own setInterval. The backend still
polls the same free yfinance quote on a short server-side loop (services/
data_service.py::get_latest_price, already TTL-cached ~8s) -- this is a
near-real-time push on top of that shared poll, not a raw broker tick feed,
since that needs a paid streaming market-data entitlement this app doesn't
have. Only a ticker's *changed* price is ever sent, not a value on every tick.
"""

import asyncio
import os
from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from services.data_service import get_latest_price
from web.backend.auth import SESSION_COOKIE_NAME, SESSION_SECRET, _is_revoked

router = APIRouter(prefix="/api/v1", tags=["price-stream"])

POLL_INTERVAL_SECONDS = 3
MAX_TICKERS = 25

# Same allow-list CORSMiddleware uses in main.py (not importable from there
# without pulling in the whole app module) -- a browser still attaches a
# SameSite=Lax session cookie to a cross-origin WebSocket handshake (CORS
# doesn't govern WebSockets the way it does fetch/XHR), so this is the one
# place in the app that has to check Origin itself rather than relying on
# the CORS middleware to have already done it.
_ALLOWED_ORIGINS = set(
    os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001").split(",")
)


async def _authenticate(websocket: WebSocket) -> dict | None:
    """The same session cookie / JWT / revocation check web.backend.auth.
    verify_bearer_token does for an HTTP request, against a WebSocket
    handshake instead. A browser's native WebSocket() call sends the
    httpOnly session cookie automatically on a same-origin connection, so
    this needs no token-in-the-URL (which would otherwise end up in nginx
    access logs and Referer headers)."""
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in _ALLOWED_ORIGINS:
        return None
    token = websocket.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    try:
        payload = jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    if await _is_revoked(payload["sub"], payload.get("iat")):
        return None
    return {"id": payload["sub"], "email": payload.get("email")}


@router.websocket("/ws/prices")
async def stream_prices(websocket: WebSocket):
    user = await _authenticate(websocket)
    if user is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()

    tickers: list[str] = []
    last_sent: dict[str, float] = {}

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                msg = None

            if isinstance(msg, dict) and isinstance(msg.get("tickers"), list):
                tickers = [
                    t.strip().upper() for t in msg["tickers"] if isinstance(t, str) and t.strip()
                ][:MAX_TICKERS]

            for ticker in list(tickers):
                price = await run_in_threadpool(get_latest_price, ticker)
                if price is not None and last_sent.get(ticker) != price:
                    last_sent[ticker] = price
                    await websocket.send_json({
                        "ticker": ticker,
                        "price": price,
                        "as_of": datetime.now(timezone.utc).isoformat(),
                    })
    except WebSocketDisconnect:
        pass
