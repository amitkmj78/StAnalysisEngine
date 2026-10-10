from datetime import date
from unittest.mock import patch

from services.daily_brief_service import _format_morning_brief, _select_news_tickers, build_evening_recap


def _row(ticker, day_gain, value_now=1000.0, is_paper=False):
    """A realistic-shaped row from compute_portfolio_performance -- real
    code always carries value_now (a number or None), day_gain_pct is
    derived the same way production data would be. is_paper is NOT set
    here -- _tag_rows_with_is_paper overwrites it from the positions list
    by index, the same as the real call path, so tests exercise that
    wiring instead of bypassing it."""
    return {"ticker": ticker, "day_gain": day_gain, "day_gain_pct": round(day_gain / (value_now - day_gain) * 100.0, 2), "value_now": value_now}


def _performance(rows):
    return {"rows": rows}


def _positions(rows, is_paper=False):
    return [{"ticker": r["ticker"], "shares": 1.0, "avg_cost": 10.0, "is_paper": is_paper} for r in rows]


def test_build_evening_recap_returns_none_with_no_positions():
    assert build_evening_recap([]) is None


def test_build_evening_recap_returns_none_when_no_row_has_day_gain():
    rows = [{"ticker": "AAA", "day_gain": None, "day_gain_pct": None, "value_now": None}]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(rows)):
        result = build_evening_recap(_positions(rows))
    assert result is None


def test_build_evening_recap_states_portfolio_and_benchmark_pct():
    rows = [_row("AAA", 50.0, value_now=1050.0), _row("BBB", -20.0, value_now=980.0)]
    # day_gain=30, value_now=2030, value_before=2000 -> +1.50%
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(rows)), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.8
    ):
        result = build_evening_recap(_positions(rows))

    assert result is not None
    assert "+1.50%" in result["text_body"]
    assert "+0.80%" in result["text_body"]
    assert "SPY" in result["text_body"]
    assert "portfolio +1.50% today" in result["subject"]
    assert "Your real holdings" in result["text_body"]
    assert "PAPER" not in result["text_body"]


def test_build_evening_recap_ranks_contributors_and_detractors():
    rows = [
        _row("GAIN_BIG", 100.0, value_now=2100.0),
        _row("GAIN_SMALL", 10.0, value_now=2010.0),
        _row("LOSS_BIG", -80.0, value_now=1920.0),
        _row("LOSS_SMALL", -5.0, value_now=1995.0),
    ]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(rows)), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.5
    ):
        result = build_evening_recap(_positions(rows))

    body = result["text_body"]
    contributors_section = body.split("Top contributors:")[1].split("Top detractors:")[0]
    detractors_section = body.split("Top detractors:")[1]

    assert contributors_section.index("GAIN_BIG") < contributors_section.index("GAIN_SMALL")
    assert detractors_section.index("LOSS_BIG") < detractors_section.index("LOSS_SMALL")


def test_build_evening_recap_caps_each_direction_at_three():
    rows = [_row(f"G{i}", float(10 - i), value_now=1000.0 + (10 - i)) for i in range(5)]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(rows)), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.3
    ):
        result = build_evening_recap(_positions(rows))

    contributors_section = result["text_body"].split("Top contributors:")[1]
    assert contributors_section.count("G") == 3
    assert "G3" not in contributors_section and "G4" not in contributors_section


def test_build_evening_recap_labels_paper_only_holdings():
    rows = [_row("PAPER1", 25.0, value_now=1025.0)]
    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(rows)), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.1
    ):
        result = build_evening_recap(_positions(rows, is_paper=True))

    assert "paper-trading portfolio" in result["subject"]
    assert "Your PAPER-TRADING holdings (practice money, not real)" in result["text_body"]
    assert "Your real holdings" not in result["text_body"]


def test_build_evening_recap_keeps_real_and_paper_totals_separate():
    """The whole point of the real/paper split: neither total is ever
    summed with the other, and the subject line doesn't collapse to one
    (misleading) figure when both are present."""
    real_rows = [_row("REAL1", 50.0, value_now=1050.0)]
    paper_rows = [_row("PAPER1", -200.0, value_now=800.0)]
    all_rows = real_rows + paper_rows
    positions = _positions(real_rows, is_paper=False) + _positions(paper_rows, is_paper=True)

    with patch("services.daily_brief_service.compute_portfolio_performance", return_value=_performance(all_rows)), patch(
        "services.daily_brief_service._benchmark_today_pct", return_value=0.2
    ):
        result = build_evening_recap(positions)

    body = result["text_body"]
    assert "Your real holdings: +$50.00 (+5.00%)" in body
    assert "Your PAPER-TRADING holdings (practice money, not real): $-200.00 (-20.00%)" in body
    # Neither a blended dollar figure nor a blended percentage appears.
    assert "$-150.00" not in body
    assert "real + paper-trading holdings today" in result["subject"]


def _rows_with_gain(tickers_and_pct, is_paper=False):
    return [
        {"ticker": t, "day_gain": pct, "day_gain_pct": pct, "value_now": 1000.0 + pct, "is_paper": is_paper}
        for t, pct in tickers_and_pct
    ]


def test_select_news_tickers_prioritizes_signal_changes_then_movers():
    rows = _rows_with_gain([("A", 1.0), ("B", 10.0), ("C", -5.0)])
    signal_changes = [{"ticker": "A", "horizon": "short", "old_signal": "Hold", "new_signal": "Buy"}]
    result = _select_news_tickers(rows, signal_changes)
    assert result == ["A", "B", "C"]  # A first (signal change), then by |day_gain_pct| descending


def test_select_news_tickers_dedupes_and_caps_at_three():
    rows = _rows_with_gain([("A", 1.0), ("B", 10.0), ("C", -5.0), ("D", 20.0)])
    signal_changes = [{"ticker": "A", "horizon": "short", "old_signal": "Hold", "new_signal": "Buy"}]
    result = _select_news_tickers(rows, signal_changes)
    assert len(result) == 3
    assert result[0] == "A"  # not duplicated even though it's also a mover


def test_format_morning_brief_returns_none_cases_render_independently():
    # No positions-with-gain, no signal changes, no earnings, no regime, no news --
    # each section should still render its own "nothing" line, not crash or vanish.
    result = _format_morning_brief(
        today=date(2026, 10, 2),
        rows_with_gain=[],
        signal_change_rows=[],
        earnings_today=[],
        regime=None,
        news_tickers=[],
        sentiment_by_ticker={},
    )
    body = result["text_body"]
    assert "No price data yet" in body
    assert "Signal Changes:\n  None today." in body
    assert "Earnings Today:\n  None today." in body
    assert "No regime reading available yet." in body
    assert "Nothing notable today." in body


def test_format_morning_brief_renders_each_section_when_present():
    result = _format_morning_brief(
        today=date(2026, 10, 2),
        rows_with_gain=_rows_with_gain([("AAA", 30.0)]),  # value_now=1030, value_before=1000 -> +3.00%
        signal_change_rows=[{"ticker": "AAA", "horizon": "short", "old_signal": "Hold", "new_signal": "Buy"}],
        earnings_today=[{"ticker": "BBB", "date": "2026-10-02", "market_timing": "before market open"}],
        regime="Risk-On",
        news_tickers=["AAA"],
        sentiment_by_ticker={"AAA": {"label": "Bullish", "reasoning": "Strong guidance."}},
    )
    body = result["text_body"]
    assert "Real portfolio: +$30.00 (+3.00%)" in body
    assert "AAA (Short-term): Hold → Buy" in body
    assert "BBB: reports before market open" in body
    assert "Risk-On" in body
    assert "AAA (Bullish): Strong guidance." in body
    assert "Morning brief for 2026-10-02" == result["subject"]


def test_format_morning_brief_separates_paper_overnight_moves():
    result = _format_morning_brief(
        today=date(2026, 10, 2),
        rows_with_gain=_rows_with_gain([("REAL1", 10.0)]) + _rows_with_gain([("PAPER1", -10.0)], is_paper=True),
        signal_change_rows=[],
        earnings_today=[],
        regime=None,
        news_tickers=[],
        sentiment_by_ticker={},
    )
    body = result["text_body"]
    assert "Real portfolio:" in body
    assert "Paper-trading portfolio (practice money, not real):" in body
    assert "REAL1" in body
    assert "PAPER1" in body


def test_format_morning_brief_news_ticker_without_sentiment_shows_fallback():
    result = _format_morning_brief(
        today=date(2026, 10, 2),
        rows_with_gain=[],
        signal_change_rows=[],
        earnings_today=[],
        regime=None,
        news_tickers=["ZZZ"],
        sentiment_by_ticker={},
    )
    assert "ZZZ: no sentiment reading available." in result["text_body"]


def test_build_morning_brief_returns_none_with_no_positions():
    import asyncio

    from services.daily_brief_service import build_morning_brief

    result = asyncio.run(build_morning_brief([], "user-1", [], []))
    assert result is None
