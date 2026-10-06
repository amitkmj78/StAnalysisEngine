"""STB-8: strategy rules written as plain English, one per line, turned into the same rules the builder runs.

Example:
    entry:
    RSI(14) crosses above 30
    Close above SMA(200)
    exit:
    RSI(14) above 70
    stop loss 8%
    wait 5 days after selling

Every line is either a rule (a condition that must be true, or any exit condition that is true), a protective setting
(stop loss, trailing stop, time stop, take profit), a wait after a sale, or a section header ("entry:" / "exit:").
A line that matches none of these is reported with its line number and an example. Nothing is guessed.
"""

import re
from typing import Optional

EXAMPLES = "RSI(14) crosses above 30, Close above SMA(200), SMA(20) crosses above SMA(50), volume vs 20-day average above 50%, within 2% of 52-week high, ATR below 2%"

OPS = {"crosses above": "crosses_above", "crosses below": "crosses_below", "above": ">", "below": "<"}
_NUM = r"(-?\d+(?:\.\d+)?)"

_RULES = [
    (re.compile(rf"^rsi\s*\(\s*14\s*\)\s+(crosses above|crosses below|above|below)\s+{_NUM}$"),
     lambda m: {"field": "rsi_14", "op": OPS[m.group(1)], "value": float(m.group(2))}),
    (re.compile(rf"^(?:close|price)\s+(crosses above|crosses below|above|below)\s+sma\s*\(\s*(50|200)\s*\)$"),
     lambda m: {"field": f"close_vs_sma_{m.group(2)}_pct", "op": OPS[m.group(1)], "value": 0.0}),
    (re.compile(r"^sma\s*\(\s*20\s*\)\s+(crosses above|crosses below|above|below)\s+sma\s*\(\s*50\s*\)$"),
     lambda m: {"field": "sma_20_vs_50_pct", "op": OPS[m.group(1)], "value": 0.0}),
    (re.compile(rf"^volume vs 20-day average (above|below)\s+{_NUM}%$"),
     lambda m: {"field": "volume_vs_20d_pct", "op": OPS[m.group(1)], "value": float(m.group(2))}),
    (re.compile(rf"^within\s+{_NUM}%\s+of 52-week high$"),
     lambda m: {"field": "dist_52w_high_pct", "op": ">", "value": -abs(float(m.group(1)))}),
    (re.compile(rf"^atr\s+(above|below)\s+{_NUM}%$"),
     lambda m: {"field": "atr_14_pct", "op": OPS[m.group(1)], "value": float(m.group(2))}),
    (re.compile(rf"^sessions since earnings\s+(at most|at least)\s+(\d+)$"),
     lambda m: {"field": "sessions_since_earnings", "op": "<=" if m.group(1) == "at most" else ">=", "value": int(m.group(2))}),
]
_STOPS = [
    (re.compile(rf"^stop loss\s+{_NUM}%$"), "stop_loss_pct", float),
    (re.compile(rf"^trailing stop\s+{_NUM}%$"), "trailing_stop_pct", float),
    (re.compile(rf"^take profit\s+{_NUM}%$"), "take_profit_pct", float),
    (re.compile(r"^time stop\s+(\d+)\s+(?:trading\s+)?days?$"), "time_stop_sessions", int),
]
_WAIT = re.compile(r"^wait\s+(\d+)\s+(?:trading\s+)?days?\s+after\s+(?:selling|a sale)$")


def parse_rule_text(text: str) -> dict:
    """Parse the rule text. Raises ValueError listing every line it could not read."""
    section = "entry"
    entry, exit_, exits, cooldown = [], [], {}, None
    problems = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"\s+", " ", raw.strip().rstrip(".").lower())
        if not line or line.startswith("#"):
            continue
        if line in ("entry:", "buy:", "buy when:", "entry"):
            section = "entry"
            continue
        if line in ("exit:", "sell:", "sell when:", "exit"):
            section = "exit"
            continue
        matched = False
        for pattern, build in _RULES:
            m = pattern.match(line)
            if m:
                (entry if section == "entry" else exit_).append(build(m))
                matched = True
                break
        if matched:
            continue
        for pattern, key, cast in _STOPS:
            m = pattern.match(line)
            if m:
                exits[key] = cast(m.group(1))
                matched = True
                break
        if matched:
            continue
        m = _WAIT.match(line)
        if m:
            cooldown = int(m.group(1))
            continue
        problems.append(f"Line {number}: \"{raw.strip()}\" is not a rule this format understands. Examples: {EXAMPLES}.")
    if problems:
        raise ValueError("\n".join(problems))
    if not entry:
        raise ValueError("Add at least one buy rule under \"entry:\".")
    result = {"entry": entry, "exit": exit_, "exits": {"stop_loss_pct": None, "trailing_stop_pct": None, "atr_stop_k": None,
                                                    "time_stop_sessions": None, "take_profit_pct": None, **exits}}
    if cooldown is not None:
        result["cooldown_sessions"] = cooldown
    return result
