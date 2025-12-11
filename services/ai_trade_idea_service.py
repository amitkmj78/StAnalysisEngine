# services/ai_trade_idea_service.py

from __future__ import annotations
import re
import numpy as np
import pandas as pd
import yfinance as yf


# ==========================================================
# Helper: pull % return from context string
# ==========================================================
def _extract_expected_return_pct(context: str | None):
    if not context:
        return None

    # Return 12.5%
    m = re.search(r"[Rr]eturn\s+(-?\d+(\.\d+)?)\s*%", context)
    if m:
        return float(m.group(1))

    # fallback: any number%
    m = re.search(r"(-?\d+(\.\d+)?)\s*%", context)
    if m:
        return float(m.group(1))

    return None


# ==========================================================
# Helper: compute market technical levels safely
# ==========================================================
def _get_market_levels(ticker: str):

    try:
        data = yf.Ticker(ticker).history(period="6mo", auto_adjust=True)
    except Exception:
        return None

    if data is None or data.empty:
        return None

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # ---------------------------
    # ATR (14)
    # ---------------------------
    prev_close = close.shift(1)
    tr = np.maximum(high - low,
                    np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = tr.rolling(14, min_periods=5).mean().iloc[-1]

    # ---------------------------
    # EMAs — using EWM (correct)
    # ---------------------------
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]

    # ---------------------------
    # Support / Resistance
    # ---------------------------
    recent = close.tail(30)
    support = recent.min()
    resistance = recent.max()

    return {
        "atr": float(atr),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "ema200": float(ema200),
        "support": float(support),
        "resistance": float(resistance),
        "current": float(close.iloc[-1]),
    }


# ==========================================================
# Main API: Generate AI Trade Idea
# ==========================================================
def generate_trade_idea(
    ticker: str,
    context: str | None = None,
    expected_return_pct: float | None = None,
):

    levels = _get_market_levels(ticker)

    if not levels:
        return (
            f"⚠ Unable to compute technicals for {ticker}. Not enough data.",
            None, None, None, None
        )

    price = levels["current"]
    atr = max(0.01, float(levels["atr"]))                   # Avoid zero ATR
    support = levels["support"]
    resistance = levels["resistance"]
    ema20, ema50, ema200 = levels["ema20"], levels["ema50"], levels["ema200"]

    # ---------------------------
    # Trend classification
    # ---------------------------
    if price > ema20 > ema50 > ema200:
        trend = "Strong Uptrend"
    elif price > ema50 > ema200:
        trend = "Uptrend"
    elif price > ema200:
        trend = "Sideways / Above LT Trend"
    else:
        trend = "Weak / Downtrend"

    # ---------------------------
    # Expected return input
    # ---------------------------
    if expected_return_pct is None:
        expected_return_pct = _extract_expected_return_pct(context or "")

    if expected_return_pct is None:
        expected_return_pct = 8.0   # default short-swing return

    # ---------------------------
    # ENTRY RANGE (ATR pullback)
    # ---------------------------
    entry_low = round(max(support, price - 0.9 * atr), 2)
    entry_high = round(price - 0.25 * atr, 2)

    # Fix inverted entry ranges
    if entry_high <= entry_low:
        entry_high = round(entry_low + 0.35 * atr, 2)

    # ---------------------------
    # STOP LOSS (ATR-based)
    # ---------------------------
    stop = round(entry_low - 1.5 * atr, 2)

    # Ensure stop < entry_low
    if stop >= entry_low:
        stop = round(entry_low * 0.97, 2)

    # ---------------------------
    # TARGET (max of resistance or expected return)
    # ---------------------------
    target_by_return = price * (1 + expected_return_pct / 100)
    target = round(max(resistance, target_by_return), 2)

    # Ensure target > price
    if target <= price:
        target = round(price * 1.03, 2)

    # ---------------------------
    # Final AI-enhanced idea text
    # ---------------------------
    idea = f"""
📌 **AI-Enhanced Trade Idea — {ticker}**

### 🔍 Trend Assessment
**{trend}**

### 🎯 Entry Zone (ATR-Adjusted Pullback)
- **${entry_low} – ${entry_high}**

### 🛑 Stop Loss (Risk Control)
- **${stop}**

### 🎯 Profit Target (7–45 days)
- **${target}**
(Expected return: ~{expected_return_pct:.1f}%)

---

### 📊 Technical Summary
- ATR(14): **{atr:.2f}**
- Support: **${support:.2f}**
- Resistance: **${resistance:.2f}**
- EMA20 / EMA50 / EMA200:  
  **{ema20:.2f} / {ema50:.2f} / {ema200:.2f}**

"""

    if context:
        idea += f"\n### 📝 Additional Context\n{context}"

    return idea, entry_low, entry_high, stop, target
