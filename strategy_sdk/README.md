# Strategy SDK (STB-7)

The same engine the no-code strategy builder uses, callable from Python. A test run here matches a run in the app
with the same rules and settings. It only tests past prices: nothing is bought or sold.

```python
from strategy_sdk import backtest

result = backtest(
    tickers=["AAPL", "MSFT", "NVDA"],
    entry=[{"field": "close_vs_sma_200_pct", "op": "crosses_above", "value": 0}],
    exit=[{"field": "close_vs_sma_200_pct", "op": "crosses_below", "value": 0}],
    trailing_stop_pct=15,
)
print(result["strategy"]["cagr_pct"], result["basket"]["cagr_pct"], result["benchmark_spy"]["cagr_pct"])
print(result["explanation"]["bottom_line"])
```

Fields: `rsi_14`, `close_vs_sma_50_pct`, `close_vs_sma_200_pct`, `sma_20_vs_50_pct`, `dist_52w_high_pct`,
`volume_vs_20d_pct`, `atr_14_pct`, `sessions_since_earnings`. Regime rules are not available in the SDK yet.

Operators: `>`, `<`, `>=`, `<=`, `crosses_above`, `crosses_below`.

Protective exits: `stop_loss_pct`, `trailing_stop_pct`, `time_stop_sessions`, `take_profit_pct`. At least one is needed
unless the caller passes `explain=False` and handles the warning themselves in the app; the engine enforces it.
