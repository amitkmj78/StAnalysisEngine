"""One-off runner for AGT-30's backtest leg (see services/agent/validation.py
for the full scope and the two disclosed narrowings). Fetches SPY's real
price history and prints the honest result -- not run automatically by
anything; re-run by hand whenever the agent's stop-loss config changes."""

import json
import sys

sys.path.insert(0, ".")

from services.agent.validation import run_agt30_stop_validation  # noqa: E402
from services.yfinance_cache import get_cached_history  # noqa: E402


def main() -> None:
    df = get_cached_history("SPY", "max", auto_adjust=True)
    if df.empty:
        print("No SPY history returned -- cannot run the validation.")
        return
    report = run_agt30_stop_validation(df)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
