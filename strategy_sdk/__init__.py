"""STB-7: a Python interface to the same strategy engine the no-code builder uses.

Rules are written as dictionaries, the same shape the builder stores:
    {"field": "rsi_14", "op": "crosses_above", "value": 30}
The price data comes from the same cache as the builder, so a test run here matches a test run in the app with the same
settings. Nothing is bought or sold: this only tests past prices.

Example:
    from strategy_sdk import backtest
    result = backtest(
        tickers=["AAPL", "MSFT", "NVDA"],
        entry=[{"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0}],
        exit=[{"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": 0}],
        trailing_stop_pct=15,
    )
    print(result["strategy"]["cagr_pct"], result["basket"]["cagr_pct"])
"""

from typing import Optional

from services.strategy_engine import DEFAULT_COOLDOWN, DEFAULT_COST_BPS, DEFAULT_SLIPPAGE_BPS, feature_frame, run_backtest
from services.strategy_explainer import explain_backtest
from services.yfinance_cache import get_cached_history

BENCHMARK = "SPY"
FIELDS = (
    "rsi_14", "close_vs_sma_50_pct", "close_vs_sma_200_pct", "sma_20_vs_50_pct",
    "dist_52w_high_pct", "volume_vs_20d_pct", "atr_14_pct", "sessions_since_earnings",
)
OPERATORS = (">", "<", ">=", "<=", "crosses_above", "crosses_below")


def backtest(
    tickers: list[str],
    entry: list[dict],
    exit: Optional[list[dict]] = None,
    stop_loss_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    time_stop_sessions: Optional[int] = None,
    take_profit_pct: Optional[float] = None,
    cooldown_sessions: int = DEFAULT_COOLDOWN,
    cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    period: str = "5y",
    benchmark: str = BENCHMARK,
    explain: bool = True,
) -> dict:
    """Run one test of the rules on the tickers. Raises ValueError with the same messages the app gives."""
    exit_rules = list(exit or [])
    exits = {
        "stop_loss_pct": stop_loss_pct, "trailing_stop_pct": trailing_stop_pct, "atr_stop_k": None,
        "time_stop_sessions": time_stop_sessions, "take_profit_pct": take_profit_pct,
    }
    frames = {}
    for ticker in tickers:
        history = get_cached_history(ticker.upper(), period, True, None)
        if history.empty or not {"Open", "High", "Low", "Close"}.issubset(history.columns):
            raise ValueError(f"No price history for {ticker}.")
        columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in history.columns]
        frames[ticker.upper()] = feature_frame(history[columns])
    bench = get_cached_history(benchmark, period, True, None)
    if bench.empty:
        raise ValueError(f"No price history for the benchmark {benchmark}.")
    result = run_backtest(
        frames, entry, exit_rules, bench["Close"], variants_tried=1, exits_raw=exits,
        cooldown=cooldown_sessions, cost_bps=cost_bps, slippage_bps=slippage_bps,
    )
    if explain:
        result["explanation"] = explain_backtest(result, {"entry": entry, "exit": exit_rules, "exits": exits})
    return result


def backtest_text(tickers: list[str], rules: str, **settings) -> dict:
    """Run a test from plain-English rules (see services/rule_text.py for the format). Extra settings, such as period or
    cost_bps, are passed on to backtest(). Settings in the text (stops, wait) apply unless overridden here."""
    from services.rule_text import parse_rule_text

    parsed = parse_rule_text(rules)
    exits = parsed["exits"]
    options = {
        "stop_loss_pct": exits.get("stop_loss_pct"), "trailing_stop_pct": exits.get("trailing_stop_pct"),
        "time_stop_sessions": exits.get("time_stop_sessions"), "take_profit_pct": exits.get("take_profit_pct"),
        "cooldown_sessions": parsed.get("cooldown_sessions", DEFAULT_COOLDOWN),
    }
    options.update(settings)
    return backtest(tickers, parsed["entry"], parsed["exit"], **options)
