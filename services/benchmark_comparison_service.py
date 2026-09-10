"""
Compares this portfolio's total return against the S&P 500 (via SPY)
since the portfolio was created -- "are we losing more than the market
overall."

Reuses compute_portfolio_performance's own total_gain_vs_cost_pct (a
real, dollar-weighted return across each position's actual cost basis)
rather than computing a second, independent return metric, so this
agrees with what "Gain vs. Paid" already shows on the Portfolio page.

The benchmark side is an approximation, not a precise dollar-weighted,
date-matched return: it's SPY's return from the PORTFOLIO's creation
date to today, not from each individual position's own purchase date
(which isn't reliably tracked for manually-entered positions). That's
the same simplification a retail investor makes eyeballing "how's the
market done since I started this portfolio" -- named as such in the
response so it isn't mistaken for something more precise.

When the portfolio is trailing by more than UNDERPERFORM_THRESHOLD_PCT,
names the worst-performing position(s) as the concrete, sized
suggestion for closing the gap -- not investment advice invented by an
LLM, just the same numbers already computed elsewhere on the page,
ranked. Same "synthesize what's already shown" boundary as
portfolio_review_service.
"""

from datetime import datetime
from typing import Optional

from .data_service import get_effective_price
from .fund_comparison_service import price_near_date
from .portfolio_performance_service import compute_portfolio_performance

BENCHMARK_TICKER = "SPY"
# A small buffer so routine day-to-day noise around dead-even doesn't
# flip this on and off -- only a real, sustained gap counts.
UNDERPERFORM_THRESHOLD_PCT = 2.0
MAX_WORST_POSITIONS = 3


def compute_benchmark_comparison(positions: list[dict], portfolio_created_at: datetime) -> dict:
    performance = compute_portfolio_performance(positions, lookback_days=30)
    portfolio_return_pct: Optional[float] = performance["total_gain_vs_cost_pct"]

    benchmark_price_then = price_near_date(BENCHMARK_TICKER, portfolio_created_at)
    benchmark_price_now = get_effective_price(BENCHMARK_TICKER)

    benchmark_return_pct: Optional[float] = None
    if benchmark_price_then and benchmark_price_now:
        benchmark_return_pct = (benchmark_price_now / benchmark_price_then - 1.0) * 100.0

    gap_pct: Optional[float] = None
    underperforming = False
    if portfolio_return_pct is not None and benchmark_return_pct is not None:
        gap_pct = portfolio_return_pct - benchmark_return_pct
        underperforming = gap_pct < -UNDERPERFORM_THRESHOLD_PCT

    suggestion: Optional[str] = None
    worst_positions: list[dict] = []
    if underperforming:
        losers = sorted(
            (
                r
                for r in performance["rows"]
                if r["gain_vs_cost_pct"] is not None and r["gain_vs_cost_pct"] < 0
            ),
            key=lambda r: r["gain_vs_cost_pct"],
        )[:MAX_WORST_POSITIONS]
        worst_positions = [
            {
                "ticker": r["ticker"],
                "gain_vs_cost_pct": r["gain_vs_cost_pct"],
                "gain_vs_cost": r["gain_vs_cost"],
                "value_now": r["value_now"],
            }
            for r in losers
        ]
        if worst_positions:
            names = ", ".join(f"{p['ticker']} ({p['gain_vs_cost_pct']:.1f}%)" for p in worst_positions)
            suggestion = (
                f"Trailing {BENCHMARK_TICKER} by {abs(gap_pct):.1f} percentage points since this portfolio "
                f"was created. Worst performers: {names} — trimming or replacing these does the most to close "
                f"the gap, since they're pulling the total return down the most."
            )
        else:
            suggestion = (
                f"Trailing {BENCHMARK_TICKER} by {abs(gap_pct):.1f} percentage points since this portfolio "
                f"was created, even though every position is individually profitable — this is about relative "
                f"strength against the market, not a loss to cut. Consider whether more of the portfolio "
                f"should be in broad-market exposure instead of stock-picking."
            )

    return {
        "benchmark_ticker": BENCHMARK_TICKER,
        "portfolio_return_pct": portfolio_return_pct,
        "benchmark_return_pct": benchmark_return_pct,
        "gap_pct": gap_pct,
        "underperforming": underperforming,
        "worst_positions": worst_positions,
        "suggestion": suggestion,
    }
