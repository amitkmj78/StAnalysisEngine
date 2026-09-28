"""
Pure, deterministic plain-English sentences for each Phase 1 scoring
factor (docs/stock-analysis-requirements.html, EXP-2) -- no LLM call,
filled directly from the real factor value/percentile. Mirrors services/
portfolio_compare_service.py's build_headline pure-string-formatting
boundary; explicitly NOT services/quant_signal_narrative_service.py or
services/prediction_narrative_service.py, which are live LLM calls and
the wrong fit here -- factor sentences must be deterministic and free.

Each function degrades gracefully to a short "not enough data" sentence
when its value is None, matching build_headline's fallback pattern.
"""

from __future__ import annotations

from typing import Optional


def momentum_sentence(return_pct: Optional[float], window_days: int, percentile: Optional[float]) -> str:
    if return_pct is None or percentile is None:
        return "Not enough price history yet to score momentum."
    verb = "up" if return_pct >= 0 else "down"
    return (
        f"Momentum: {verb} {abs(return_pct):.1f}% over the trailing {window_days} trading days "
        f"— {percentile:.0f}th percentile in the universe."
    )


def reversal_sentence(rsi: Optional[float], percentile: Optional[float]) -> str:
    if rsi is None or percentile is None:
        return "Not enough price history yet to score short-term reversal."
    state = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    return (
        f"Short-term reversal: RSI at {rsi:.0f} ({state}) — {percentile:.0f}th percentile "
        f"for reversal potential in the universe."
    )


def value_sentence(forward_pe: Optional[float], percentile: Optional[float]) -> str:
    if forward_pe is None or percentile is None:
        return "No forward P/E on record yet to score value."
    return f"Value: forward P/E of {forward_pe:.1f} — {percentile:.0f}th percentile (cheaper is higher) in the universe."


def growth_sentence(revenue_growth_pct: Optional[float], earnings_growth_pct: Optional[float], percentile: Optional[float]) -> str:
    if percentile is None:
        return "Not enough fundamentals data yet to score growth."
    parts = []
    if revenue_growth_pct is not None:
        parts.append(f"revenue growth of {revenue_growth_pct:.1f}%")
    if earnings_growth_pct is not None:
        parts.append(f"earnings growth of {earnings_growth_pct:.1f}%")
    detail = " and ".join(parts) if parts else "growth data"
    return f"Growth: {detail} — {percentile:.0f}th percentile in the universe."


def low_vol_sentence(annualized_volatility_pct: Optional[float], percentile: Optional[float]) -> str:
    if annualized_volatility_pct is None or percentile is None:
        return "Not enough price history yet to score volatility."
    return (
        f"Low volatility: {annualized_volatility_pct:.1f}% annualized — {percentile:.0f}th percentile "
        f"(calmer is higher) in the universe."
    )
