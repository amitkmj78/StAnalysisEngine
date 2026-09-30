from datetime import date

import numpy as np
import pandas as pd
import pytest

from services.stock_detail_service import (
    evaluate_signal_history,
    evaluate_signal_outcome,
    next_day_move_pct,
    next_earnings_date,
    past_earnings_dates,
    recent_dividends,
    select_peers,
    typical_earnings_move,
    upcoming_earnings_in_window,
)


def _universe():
    return pd.DataFrame(
        [
            {"Ticker": "AAPL", "Name": "Apple", "GICS Sector": "Information Technology", "Market Cap ($B)": 3000.0},
            {"Ticker": "MSFT", "Name": "Microsoft", "GICS Sector": "Information Technology", "Market Cap ($B)": 2900.0},
            {"Ticker": "NVDA", "Name": "Nvidia", "GICS Sector": "Information Technology", "Market Cap ($B)": 3200.0},
            {"Ticker": "ORCL", "Name": "Oracle", "GICS Sector": "Information Technology", "Market Cap ($B)": 500.0},
            {"Ticker": "CRM", "Name": "Salesforce", "GICS Sector": "Information Technology", "Market Cap ($B)": 250.0},
            {"Ticker": "ADBE", "Name": "Adobe", "GICS Sector": "Information Technology", "Market Cap ($B)": 200.0},
            {"Ticker": "JNJ", "Name": "Johnson & Johnson", "GICS Sector": "Health Care", "Market Cap ($B)": 400.0},
        ]
    )


def test_select_peers_prefers_closest_market_cap_within_same_sector():
    peers = select_peers("AAPL", _universe(), top_n=3)
    tickers = [p["ticker"] for p in peers]
    # AAPL=3000: closest by |cap - 3000| among same-sector, non-self tickers
    # is MSFT(2900, dist 100), NVDA(3200, dist 200), ORCL(500, dist 2500).
    assert tickers == ["MSFT", "NVDA", "ORCL"]


def test_select_peers_excludes_other_sectors():
    peers = select_peers("AAPL", _universe(), top_n=10)
    assert "JNJ" not in [p["ticker"] for p in peers]


def test_select_peers_unknown_ticker_returns_empty():
    assert select_peers("ZZZZ", _universe()) == []


def test_select_peers_missing_market_cap_returns_empty():
    df = pd.DataFrame([{"Ticker": "XYZ", "Name": "X", "GICS Sector": "Energy", "Market Cap ($B)": np.nan}])
    assert select_peers("XYZ", df) == []


def test_next_earnings_date_picks_the_soonest_upcoming_row():
    earnings = pd.DataFrame(
        {"EPS Estimate": [1.5, 1.8, None]},
        index=pd.to_datetime(["2026-01-01", "2026-03-15", "2026-06-15"]),
    )
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": 1.8}


def test_next_earnings_date_none_when_no_future_row():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-01-01"]))
    assert next_earnings_date(earnings, as_of=date(2026, 2, 1)) is None


def test_next_earnings_date_empty_input():
    assert next_earnings_date(pd.DataFrame()) is None


def test_next_earnings_date_handles_tz_aware_index_without_crashing():
    earnings = pd.DataFrame(
        {"EPS Estimate": [2.0]},
        index=pd.date_range("2026-03-15", periods=1, tz="America/New_York"),
    )
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": 2.0}


def test_next_earnings_date_none_eps_estimate_when_not_provided():
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.to_datetime(["2026-03-15"]))
    result = next_earnings_date(earnings, as_of=date(2026, 2, 1))
    assert result == {"date": "2026-03-15", "eps_estimate": None}


def test_past_earnings_dates_returns_only_dates_before_as_of_most_recent_first():
    earnings = pd.DataFrame(
        {"Reported EPS": [1.1, 1.3, None], "EPS Estimate": [1.0, 1.2, 1.4], "Surprise(%)": [10.0, 8.3, None]},
        index=pd.to_datetime(["2025-09-15", "2025-12-15", "2026-06-15"]),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 2, 1))
    assert result == [
        {
            "date": "2025-12-15", "reported_eps": 1.3, "eps_estimate": 1.2, "eps_beat": True,
            "surprise_pct": 8.3, "revenue_beat": None,
        },
        {
            "date": "2025-09-15", "reported_eps": 1.1, "eps_estimate": 1.0, "eps_beat": True,
            "surprise_pct": 10.0, "revenue_beat": None,
        },
    ]


def test_past_earnings_dates_eps_beat_false_on_miss():
    earnings = pd.DataFrame(
        {"Reported EPS": [0.9], "EPS Estimate": [1.0], "Surprise(%)": [-10.0]},
        index=pd.to_datetime(["2025-12-15"]),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 1, 1))
    assert result[0]["eps_beat"] is False


def test_past_earnings_dates_eps_beat_none_when_estimate_missing():
    earnings = pd.DataFrame(
        {"Reported EPS": [1.1], "EPS Estimate": [None], "Surprise(%)": [None]},
        index=pd.to_datetime(["2025-12-15"]),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 1, 1))
    assert result[0]["eps_beat"] is None


def test_past_earnings_dates_revenue_beat_always_none():
    # Locks in the deliberate "honest gap" decision -- yfinance has no
    # historical revenue-estimate-vs-actual for past quarters anywhere,
    # so this must never silently start returning True/False.
    earnings = pd.DataFrame(
        {"Reported EPS": [1.1, 0.9], "EPS Estimate": [1.0, 1.0], "Surprise(%)": [10.0, -10.0]},
        index=pd.to_datetime(["2025-09-15", "2025-12-15"]),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 1, 1))
    assert all(row["revenue_beat"] is None for row in result)


def test_past_earnings_dates_respects_limit():
    earnings = pd.DataFrame(
        {"Reported EPS": [1.0, 1.1, 1.2]},
        index=pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01"]),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 1, 1), limit=2)
    assert len(result) == 2
    assert result[0]["date"] == "2025-07-01"


def test_past_earnings_dates_empty_input():
    assert past_earnings_dates(pd.DataFrame()) == []


def test_past_earnings_dates_handles_tz_aware_index_without_crashing():
    earnings = pd.DataFrame(
        {"Reported EPS": [2.0]},
        index=pd.date_range("2025-03-15", periods=1, tz="America/New_York"),
    )
    result = past_earnings_dates(earnings, as_of=date(2026, 2, 1))
    assert result[0]["date"] == "2025-03-15"
    assert result[0]["reported_eps"] == 2.0


def test_recent_dividends_most_recent_first():
    dividends = pd.Series([0.20, 0.22, 0.24], index=pd.to_datetime(["2026-01-01", "2026-04-01", "2026-07-01"]))
    result = recent_dividends(dividends, top_n=2)
    assert result == [{"date": "2026-07-01", "amount": 0.24}, {"date": "2026-04-01", "amount": 0.22}]


def test_recent_dividends_empty_series():
    assert recent_dividends(pd.Series(dtype=float)) == []


def _closes(prices, start="2026-01-01"):
    return pd.Series(prices, index=pd.bdate_range(start, periods=len(prices)))


def test_evaluate_signal_outcome_buy_hit_when_price_rises():
    closes = _closes([100, 101, 102, 103, 104, 105])
    result = evaluate_signal_outcome(date(2026, 1, 1), "Buy", closes, horizon_days=3)
    assert result["outcome"] == "hit"
    assert result["realized_return_pct"] > 0


def test_evaluate_signal_outcome_buy_miss_when_price_falls():
    closes = _closes([100, 99, 98, 97, 96, 95])
    result = evaluate_signal_outcome(date(2026, 1, 1), "Buy", closes, horizon_days=3)
    assert result["outcome"] == "miss"
    assert result["realized_return_pct"] < 0


def test_evaluate_signal_outcome_trim_hit_when_price_falls():
    closes = _closes([100, 99, 98, 97, 96, 95])
    result = evaluate_signal_outcome(date(2026, 1, 1), "Trim", closes, horizon_days=3)
    assert result["outcome"] == "hit"


def test_evaluate_signal_outcome_trim_miss_when_price_rises():
    closes = _closes([100, 101, 102, 103, 104, 105])
    result = evaluate_signal_outcome(date(2026, 1, 1), "Trim", closes, horizon_days=3)
    assert result["outcome"] == "miss"


def test_evaluate_signal_outcome_hold_has_no_verdict_but_reports_return():
    closes = _closes([100, 101, 102, 103, 104, 105])
    result = evaluate_signal_outcome(date(2026, 1, 1), "Hold", closes, horizon_days=3)
    assert result["outcome"] is None
    assert result["realized_return_pct"] > 0


def test_evaluate_signal_outcome_none_when_horizon_not_elapsed():
    closes = _closes([100, 101, 102])
    assert evaluate_signal_outcome(date(2026, 1, 1), "Buy", closes, horizon_days=5) is None


def test_evaluate_signal_outcome_none_when_as_of_date_past_series_end():
    closes = _closes([100, 101, 102])
    assert evaluate_signal_outcome(date(2026, 6, 1), "Buy", closes, horizon_days=1) is None


def test_evaluate_signal_outcome_none_for_empty_series():
    assert evaluate_signal_outcome(date(2026, 1, 1), "Buy", pd.Series(dtype=float), horizon_days=1) is None


def test_evaluate_signal_outcome_handles_tz_aware_index_without_crashing():
    # Regression: yfinance's real history index is tz-aware
    # (America/New_York); comparing it against a tz-naive pd.Timestamp
    # raised TypeError in production even though synthetic tz-naive test
    # data never caught it.
    closes = pd.Series(
        [100, 101, 102, 103, 104, 105],
        index=pd.date_range("2026-01-01", periods=6, freq="B", tz="America/New_York"),
    )
    result = evaluate_signal_outcome(date(2026, 1, 1), "Buy", closes, horizon_days=3)
    assert result["outcome"] == "hit"


def test_evaluate_signal_history_matures_short_but_not_long_horizon():
    closes = _closes([100 + i for i in range(15)])  # 15 trading days, rising
    history = [
        {
            "as_of_date": "2026-01-01",
            "short_score": 75.0,
            "short_signal": "Buy",
            "long_score": 55.0,
            "long_signal": "Hold",
        }
    ]
    result = evaluate_signal_history(history, closes)
    assert len(result) == 1
    row = result[0]
    assert row["short_score"] == 75.0  # original fields preserved
    assert row["short_outcome"] is not None
    assert row["short_outcome"]["outcome"] == "hit"
    assert row["long_outcome"] is None  # 252-trading-day horizon can't mature in a 15-day series


# ---------------------------------------------------------------------------
# next_day_move_pct / typical_earnings_move / upcoming_earnings_in_window
# (ERN-1/2/3). 2026-01-02 is a Friday, so bdate_range from there gives
# Fri, Mon, Tue, Wed, Thu -- five consecutive trading days spanning a
# weekend, the same shape real daily bars have.
# ---------------------------------------------------------------------------


def test_next_day_move_pct_amc_uses_entry_and_next_day_exit():
    # AMC (16:00) on 2026-01-05 (Mon, index 1): entry=that day's close,
    # exit=2026-01-06's close.
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-05 16:00:00")]))
    closes = _closes([100, 100, 105, 106, 107], start="2026-01-02")
    result = next_day_move_pct(earnings, closes, as_of=date(2026, 2, 1))
    assert len(result) == 1
    assert result[0]["market_timing"] == "after_market"
    assert result[0]["move_pct"] == pytest.approx(5.0)


def test_next_day_move_pct_bmo_uses_prior_day_entry_and_earnings_day_exit():
    # BMO (07:00) on 2026-01-06 (Tue, index 2): entry=prior day
    # (2026-01-05)'s close, exit=that day's own close.
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-06 07:00:00")]))
    closes = _closes([100, 102, 110, 106, 107], start="2026-01-02")
    result = next_day_move_pct(earnings, closes, as_of=date(2026, 2, 1))
    assert result[0]["market_timing"] == "before_market"
    assert result[0]["move_pct"] == pytest.approx((110 / 102 - 1) * 100, abs=0.01)


def test_next_day_move_pct_none_when_price_data_insufficient():
    # AMC on the very last day in the closes series -- no next-day bar
    # exists yet, so move_pct must be None, not raise.
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-08 16:00:00")]))
    closes = _closes([100, 100, 105, 106, 107], start="2026-01-02")
    result = next_day_move_pct(earnings, closes, as_of=date(2026, 2, 1))
    assert result[0]["move_pct"] is None


def test_next_day_move_pct_handles_tz_aware_earnings_index():
    earnings = pd.DataFrame(
        {"EPS Estimate": [None]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-05 16:00:00")]).tz_localize("America/New_York"),
    )
    closes = _closes([100, 100, 105, 106, 107], start="2026-01-02")
    result = next_day_move_pct(earnings, closes, as_of=date(2026, 2, 1))
    assert result[0]["market_timing"] == "after_market"
    assert result[0]["move_pct"] == pytest.approx(5.0)


def test_next_day_move_pct_handles_tz_aware_closes_index():
    # Mirrors get_cached_history's real, tz-aware (America/New_York) index.
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-05 16:00:00")]))
    closes = pd.Series(
        [100, 100, 105, 106, 107], index=pd.bdate_range("2026-01-02", periods=5, tz="America/New_York")
    )
    result = next_day_move_pct(earnings, closes, as_of=date(2026, 2, 1))
    assert result[0]["move_pct"] == pytest.approx(5.0)


def test_next_day_move_pct_empty_earnings():
    assert next_day_move_pct(pd.DataFrame(), _closes([100, 101]), as_of=date(2026, 1, 1)) == []


def test_next_day_move_pct_empty_closes():
    earnings = pd.DataFrame({"EPS Estimate": [None]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-05 16:00:00")]))
    assert next_day_move_pct(earnings, pd.Series(dtype=float), as_of=date(2026, 1, 1)) == []


def test_typical_earnings_move_averages_absolute_values():
    # A +5% quarter and a -5% quarter both mean "usually moves ~5%" --
    # direction must not cancel out.
    moves = [{"move_pct": 5.0}, {"move_pct": -5.0}, {"move_pct": 3.0}]
    result = typical_earnings_move(moves)
    assert result["avg_abs_move_pct"] == round(13 / 3, 1)
    assert result["quarters_counted"] == 3


def test_typical_earnings_move_ignores_none_rows_and_counts_correctly():
    moves = [{"move_pct": 5.0}, {"move_pct": None}, {"move_pct": 3.0}]
    result = typical_earnings_move(moves)
    assert result["quarters_counted"] == 2
    assert result["avg_abs_move_pct"] == pytest.approx(4.0)


def test_typical_earnings_move_none_when_no_computable_quarters():
    assert typical_earnings_move([{"move_pct": None}, {"move_pct": None}]) is None


def test_typical_earnings_move_empty_list():
    assert typical_earnings_move([]) is None


def test_upcoming_earnings_in_window_within_30_days():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-02-10"]))
    result = upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1), window_days=30)
    assert result["date"] == "2026-02-10"
    assert result["eps_estimate"] == 1.5


def test_upcoming_earnings_in_window_excludes_beyond_window():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-04-01"]))
    assert upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1), window_days=30) is None


def test_upcoming_earnings_in_window_excludes_past_dates():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-01-15"]))
    assert upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1), window_days=30) is None


def test_upcoming_earnings_in_window_infers_before_market():
    earnings = pd.DataFrame(
        {"EPS Estimate": [1.5]}, index=pd.DatetimeIndex([pd.Timestamp("2026-02-10 07:00:00")])
    )
    result = upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1))
    assert result["market_timing"] == "before_market"


def test_upcoming_earnings_in_window_infers_after_market():
    earnings = pd.DataFrame(
        {"EPS Estimate": [1.5]}, index=pd.DatetimeIndex([pd.Timestamp("2026-02-10 16:00:00")])
    )
    result = upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1))
    assert result["market_timing"] == "after_market"


def test_upcoming_earnings_in_window_empty_input():
    assert upcoming_earnings_in_window(pd.DataFrame()) is None


def test_upcoming_earnings_in_window_custom_window_days():
    earnings = pd.DataFrame({"EPS Estimate": [1.5]}, index=pd.to_datetime(["2026-02-08"]))  # 7 days out
    assert upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1), window_days=5) is None
    assert upcoming_earnings_in_window(earnings, as_of=date(2026, 2, 1), window_days=10) is not None
