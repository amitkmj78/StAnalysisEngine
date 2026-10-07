"""NFR-2: the closest practical equivalent of "enforced by tests" for a
live-dependency-heavy app -- a real pytest can't cheaply spin up the full
stack + real yfinance data in CI, so this hits a locally running backend
for real and reports p50/p95 against the spec's own budgets:
  - Stock detail page: p95 under 1.5s
  - Screener results: under 1s for the full universe

Usage:
    ./venv/Scripts/python.exe scripts/benchmark_pages.py [--base-url URL] [--n N] [--token TOKEN]

--token is only needed for the screener check (GET /stock-finder/rank
requires a signed-in session); without one, that check is skipped with a
clear note rather than failing. The stock-detail check needs no auth
(GET /stock/{ticker}/detail is a public route) and always runs.

Exits non-zero if any measured p95 exceeds its budget, so this can be
wired into a pre-deploy check later if desired -- not wired into
anything today, run by hand.
"""

import argparse
import statistics
import sys
import time

import httpx

STOCK_DETAIL_BUDGET_SECONDS = 1.5
SCREENER_BUDGET_SECONDS = 1.0  # spec's own bar: "under 1s for the full universe"
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def _timed_get(client: httpx.Client, url: str, **kwargs) -> float:
    start = time.perf_counter()
    resp = client.get(url, **kwargs)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return elapsed


def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] + (values[c] - values[f]) * (k - f)


def _report(label: str, elapsed: list[float], budget: float) -> bool:
    p50, p95 = _percentile(elapsed, 0.50), _percentile(elapsed, 0.95)
    passed = p95 <= budget
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] {label}: p50={p50*1000:.0f}ms p95={p95*1000:.0f}ms "
        f"(budget p95<={budget*1000:.0f}ms, n={len(elapsed)}, min={min(elapsed)*1000:.0f}ms max={max(elapsed)*1000:.0f}ms)"
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8010")
    parser.add_argument("--n", type=int, default=10, help="requests per ticker for the stock-detail check")
    parser.add_argument("--token", default=None, help="bearer token for the screener check (requires a signed-in session)")
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    args = parser.parse_args()

    all_passed = True
    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        # --- Stock detail page: p95 under 1.5s, no auth needed ---
        detail_times: list[float] = []
        for ticker in args.tickers:
            for _ in range(args.n):
                try:
                    detail_times.append(_timed_get(client, f"/api/v1/stock/{ticker}/detail"))
                except httpx.HTTPError as e:
                    print(f"[ERROR] /stock/{ticker}/detail: {e}")
        if detail_times:
            all_passed &= _report("Stock detail (/stock/{ticker}/detail)", detail_times, STOCK_DETAIL_BUDGET_SECONDS)
        else:
            print("[SKIP] Stock detail: no successful requests (is the backend running at", args.base_url, "?)")
            all_passed = False

        # --- Screener: under 1s for the full universe, needs auth ---
        if args.token:
            headers = {"Authorization": f"Bearer {args.token}"}
            screener_times: list[float] = []
            for _ in range(args.n):
                try:
                    screener_times.append(
                        _timed_get(client, "/api/v1/stock-finder/rank", params={"goal": "Growth", "universe": "All"}, headers=headers)
                    )
                except httpx.HTTPError as e:
                    print(f"[ERROR] /stock-finder/rank: {e}")
            if screener_times:
                all_passed &= _report("Screener (/stock-finder/rank, universe=All)", screener_times, SCREENER_BUDGET_SECONDS)
                print(
                    "  Note: only the FIRST request per universe per hour pays the full live-scan cost "
                    "(services/stock_finder_service.py's 1h cache) -- this budget is realistically only "
                    "met while that cache is warm, not on every single cold request."
                )
        else:
            print("[SKIP] Screener check: no --token provided (GET /stock-finder/rank requires a signed-in session).")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
