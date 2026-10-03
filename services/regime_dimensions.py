"""Regime banner dimensions (REG-4 to REG-9): display-only, deterministic.

Pure functions over a daily DataFrame. None of these feed the existing
internals score, the regime label, or the trading agent's caps. Each
reading returns its numbers, its -1/0/+1 score where the spec gives one,
and a sentence built from a fixed template that quotes the real figures,
the window used, and the rule that produced the score.
"""

from typing import Optional

import pandas as pd

RATES_WINDOW_SESSIONS = 63  # about three months of trading days
RATES_RISE_PTS = 0.50
MOVE_STRESS = 120.0
MOVE_CALM = 90.0
CREDIT_MA_SESSIONS = 50
RISK_APPETITE_SESSIONS = 63
DIVERGENCE_BREADTH_PCT = 35.0


def _last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _change(series: pd.Series, sessions: int) -> Optional[float]:
    s = series.dropna()
    if len(s) <= sessions:
        return None
    return float(s.iloc[-1] - s.iloc[-1 - sessions])


def rates_reading(df: pd.DataFrame) -> dict:
    """REG-4. -1 if the 3-month yield rise exceeds 0.50 pts or MOVE is above 120;
    +1 if yields fell and MOVE is below 90; otherwise 0."""
    yield_now = _last(df["tnx"]) if "tnx" in df else None
    move = _last(df["move"]) if "move" in df else None
    change = _change(df["tnx"], RATES_WINDOW_SESSIONS) if "tnx" in df else None
    if yield_now is None or change is None or move is None:
        return {"score": None, "yield_now": yield_now, "yield_change_pts": change, "move": move,
                "text": "Rates reading unavailable: yield or MOVE history is missing."}
    if change > RATES_RISE_PTS or move > MOVE_STRESS:
        score = -1
    elif change < 0 and move < MOVE_CALM:
        score = 1
    else:
        score = 0
    direction = "up" if change >= 0 else "down"
    text = (f"10-year yield {direction} {abs(change):.2f} pts over 3 months to {yield_now:.2f}%; "
            f"MOVE {move:.0f} (above {MOVE_STRESS:.0f} is a stress flag). Rates score {score:+d}.")
    return {"score": score, "yield_now": round(yield_now, 2), "yield_change_pts": round(change, 2),
            "move": round(move, 1), "text": text}


def credit_reading(df: pd.DataFrame) -> dict:
    """REG-5. HYG/IEF versus its 50-session average: +1 above, -1 below, 0 equal."""
    if "hyg_ief" not in df or df["hyg_ief"].dropna().shape[0] < CREDIT_MA_SESSIONS:
        return {"score": None, "ratio": None, "average_50d": None,
                "text": "Credit reading unavailable: not enough HYG/IEF history for a 50-day average."}
    ratio = float(df["hyg_ief"].dropna().iloc[-1])
    average = float(df["hyg_ief"].dropna().tail(CREDIT_MA_SESSIONS).mean())
    r6, a6 = round(ratio, 6), round(average, 6)
    score = 1 if r6 > a6 else -1 if r6 < a6 else 0
    relation = "above" if score > 0 else "below" if score < 0 else "equal to"
    text = (f"HYG/IEF ratio {ratio:.3f} is {relation} its 50-day average of {average:.3f}. "
            f"Credit scores {score:+d}.")
    return {"score": score, "ratio": round(ratio, 4), "average_50d": round(average, 4), "text": text}


def breadth_reading(df: pd.DataFrame) -> dict:
    """REG-6. SPY against its 50- and 200-session averages, and the share of S&P
    500 stocks above their own 50- and 200-day averages."""
    spy = df["spy_close"].dropna() if "spy_close" in df else pd.Series(dtype=float)
    spy_now = _last(spy)
    spy_50 = float(spy.tail(50).mean()) if len(spy) >= 50 else None
    spy_200 = float(spy.tail(200).mean()) if len(spy) >= 200 else None
    b50 = _last(df["breadth_50dma"]) if "breadth_50dma" in df else None
    b200 = _last(df["breadth_200dma"]) if "breadth_200dma" in df else None

    def _side(level, avg, label):
        if level is None or avg is None:
            return f"SPY vs its {label} average: not enough history."
        side = "above" if level > avg else "below"
        return f"SPY {side} its {label} average ({level:.2f} vs {avg:.2f})"

    parts = [_side(spy_now, spy_50, "50-day"), _side(spy_now, spy_200, "200-day")]
    if b50 is not None and b200 is not None:
        parts.append(f"{b50:.1f}% of S&P 500 stocks are above their 50-day average and {b200:.1f}% above their 200-day")
    return {
        "spy_close": round(spy_now, 2) if spy_now is not None else None,
        "spy_50dma": round(spy_50, 2) if spy_50 is not None else None,
        "spy_200dma": round(spy_200, 2) if spy_200 is not None else None,
        "spy_above_50dma": (spy_now > spy_50) if spy_now is not None and spy_50 is not None else None,
        "spy_above_200dma": (spy_now > spy_200) if spy_now is not None and spy_200 is not None else None,
        "pct_above_50dma": round(b50, 2) if b50 is not None else None,
        "pct_above_200dma": round(b200, 2) if b200 is not None else None,
        "text": "; ".join(parts) + ".",
    }


def divergence_flag(df: pd.DataFrame) -> dict:
    """REG-7. Flag when SPY is above its 50-session average but fewer than 35%
    of S&P 500 stocks are above their own 50-day average."""
    spy = df["spy_close"].dropna() if "spy_close" in df else pd.Series(dtype=float)
    b50 = _last(df["breadth_50dma"]) if "breadth_50dma" in df else None
    if len(spy) < 50 or b50 is None:
        return {"flag": None, "text": "Divergence check unavailable: not enough history."}
    spy_now = float(spy.iloc[-1])
    spy_50 = float(spy.tail(50).mean())
    flag = spy_now > spy_50 and b50 < DIVERGENCE_BREADTH_PCT
    if flag:
        text = (f"Divergence: SPY is above its 50-day average, but only {b50:.1f}% of S&P 500 stocks are above "
                f"theirs (flag below {DIVERGENCE_BREADTH_PCT:.0f}%).")
    else:
        text = (f"No divergence: {b50:.1f}% of S&P 500 stocks are above their 50-day average "
                f"(flag only when SPY is above its 50-day average and that share is below {DIVERGENCE_BREADTH_PCT:.0f}%).")
    return {"flag": bool(flag), "pct_above_50dma": round(b50, 2), "text": text}


def risk_appetite_reading(df: pd.DataFrame) -> dict:
    """REG-8. Equal-weight S&P 500 (RSP) against SPY, change over three months.
    The ratio and window are named in the sentence, per the spec."""
    change = _change(df["rsp_spy"], RISK_APPETITE_SESSIONS) if "rsp_spy" in df else None
    if change is None:
        return {"change_pct": None, "text": "Risk appetite reading unavailable: not enough RSP/SPY history."}
    start = float(df["rsp_spy"].dropna().iloc[-1 - RISK_APPETITE_SESSIONS])
    pct = (change / start) * 100.0
    text = f"Equal-weight vs S&P 500 (RSP/SPY), last 3 months: {pct:+.1f}%."
    return {"change_pct": round(pct, 2), "text": text}


def regime_dimensions(df: pd.DataFrame) -> dict:
    """All banner dimensions for the latest date. Missing inputs yield None
    readings, never a guessed value."""
    return {
        "rates": rates_reading(df),
        "credit": credit_reading(df),
        "breadth": breadth_reading(df),
        "divergence": divergence_flag(df),
        "risk_appetite": risk_appetite_reading(df),
    }
