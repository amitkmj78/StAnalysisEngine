"""BRF-1/BRF-2: daily brief email content-builders.

Both briefs are pure content-builders with no delivery logic of their
own -- the scheduler job that calls these hands the result straight to
notification_dispatcher.dispatch_alert (ticker=None, alert_type=
"morning_brief"/"evening_recap"), the same funnel every other alert
already goes through, so quiet hours/digest/channel preference is
honored for free rather than reimplemented here.
"""

from typing import Optional

from services.benchmark_comparison_service import BENCHMARK_TICKER
from services.data_service import get_effective_price, get_previous_close
from services.email_service import APP_URL
from services.portfolio_performance_service import compute_portfolio_performance

MAX_HOLDINGS_PER_DIRECTION = 3


def _fmt_dollars(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _benchmark_today_pct(ticker: str = BENCHMARK_TICKER) -> Optional[float]:
    """Same day-P&L math as a position's own day_gain_pct (previous close
    vs. the current effective price) -- not the heavier since-inception
    compute_benchmark_comparison, which this brief has no use for."""
    price_now = get_effective_price(ticker)
    prev_close = get_previous_close(ticker)
    if price_now is None or prev_close is None or prev_close == 0:
        return None
    return (price_now / prev_close - 1.0) * 100.0


def build_evening_recap(positions: list[dict]) -> Optional[dict]:
    """BRF-2: today's portfolio move next to SPY's own today move, plus
    the day's top contributors/detractors by holding. Returns None when
    there are no positions, or when no holding has a day_gain yet (e.g.
    before the first regular-session close, or a weekend/holiday run) --
    an empty brief isn't worth sending.
    """
    if not positions:
        return None

    performance = compute_portfolio_performance(positions)
    rows_with_gain = [r for r in performance["rows"] if r["day_gain"] is not None]
    if not rows_with_gain:
        return None

    total_day_gain = performance["total_day_gain"]
    total_day_gain_pct = performance["total_day_gain_pct"]
    benchmark_today_pct = _benchmark_today_pct()

    gainers = sorted(
        (r for r in rows_with_gain if r["day_gain"] > 0), key=lambda r: r["day_gain"], reverse=True
    )[:MAX_HOLDINGS_PER_DIRECTION]
    losers = sorted(
        (r for r in rows_with_gain if r["day_gain"] < 0), key=lambda r: r["day_gain"]
    )[:MAX_HOLDINGS_PER_DIRECTION]

    subject = f"Evening recap: portfolio {_fmt_pct(total_day_gain_pct)} today"

    lines = [
        f"Your portfolio: {_fmt_dollars(total_day_gain)} ({_fmt_pct(total_day_gain_pct)}) today.",
        f"{BENCHMARK_TICKER}: {_fmt_pct(benchmark_today_pct)} today.",
        "",
    ]
    if gainers:
        lines.append("Top contributors:")
        for r in gainers:
            lines.append(f"  {r['ticker']}: {_fmt_dollars(r['day_gain'])} ({_fmt_pct(r['day_gain_pct'])})")
        lines.append("")
    if losers:
        lines.append("Top detractors:")
        for r in losers:
            lines.append(f"  {r['ticker']}: {_fmt_dollars(r['day_gain'])} ({_fmt_pct(r['day_gain_pct'])})")
        lines.append("")
    lines.append(f"See {APP_URL}/portfolio for the full breakdown.")

    return {"subject": subject, "text_body": "\n".join(lines)}
