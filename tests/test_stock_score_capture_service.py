from datetime import date

import pandas as pd

from services.stock_score_capture_service import (
    RSI_MIN_ROWS,
    VOLATILITY_TRADING_DAYS,
    _annualized_volatility,
    _compute_rsi,
    _eps_revision_from_trend,
    _latest_eps_surprise,
    _momentum_and_reversal_from_rows,
    _volatility_from_rows,
    blend_growth,
    blend_quality,
)


def _pit_rows(ticker: str, closes: list[float], start=date(2026, 1, 1)):
    return [{"price_date": date.fromordinal(start.toordinal() + i), "close": c} for i, c in enumerate(closes)]


def test_momentum_stays_pit_sourced_when_enough_history():
    pit_rows_by_ticker = {"AAPL": _pit_rows("AAPL", [100.0 + i for i in range(35)])}
    result = _momentum_and_reversal_from_rows(
        ["AAPL"], pit_rows_by_ticker,
        live_momentum_fn=lambda t: (_ for _ in ()).throw(AssertionError("should not hit live fallback")),
        live_reversal_fn=lambda t: (_ for _ in ()).throw(AssertionError("should not hit live fallback")),
    )
    assert result["AAPL"]["momentum"]["source"] == "pit"
    assert result["AAPL"]["reversal"]["source"] == "pit"


def test_momentum_falls_back_to_live_with_too_little_pit_history():
    # Only 5 rows -- well short of MOMENTUM_LOOKBACK_DAYS(30)+1 and RSI_MIN_ROWS(15).
    pit_rows_by_ticker = {"NEWCO": _pit_rows("NEWCO", [10.0, 11.0, 12.0, 11.5, 13.0])}
    calls = []

    def live_momentum(t):
        calls.append(("momentum", t))
        return {"raw": 4.2, "source": "live"}

    def live_reversal(t):
        calls.append(("reversal", t))
        return {"raw": 55.0, "source": "live"}

    result = _momentum_and_reversal_from_rows(["NEWCO"], pit_rows_by_ticker, live_momentum, live_reversal)
    assert result["NEWCO"]["momentum"] == {"raw": 4.2, "source": "live"}
    assert result["NEWCO"]["reversal"] == {"raw": 55.0, "source": "live"}
    assert ("momentum", "NEWCO") in calls
    assert ("reversal", "NEWCO") in calls


def test_reversal_can_be_pit_sourced_while_momentum_falls_back_to_live():
    # Exactly RSI_MIN_ROWS(15) rows -- enough for RSI, not enough for
    # momentum's 31-row requirement -- each factor must fall back
    # independently, not as an all-or-nothing pair.
    assert RSI_MIN_ROWS == 15
    pit_rows_by_ticker = {"T": _pit_rows("T", [100.0 + i for i in range(15)])}
    result = _momentum_and_reversal_from_rows(
        ["T"], pit_rows_by_ticker,
        live_momentum_fn=lambda t: {"raw": 1.0, "source": "live"},
        live_reversal_fn=lambda t: (_ for _ in ()).throw(AssertionError("reversal should stay PIT-sourced")),
    )
    assert result["T"]["momentum"]["source"] == "live"
    assert result["T"]["reversal"]["source"] == "pit"


def test_missing_ticker_uses_live_for_both_factors():
    result = _momentum_and_reversal_from_rows(
        ["GHOST"], {},
        live_momentum_fn=lambda t: {"raw": None, "source": "live"},
        live_reversal_fn=lambda t: {"raw": None, "source": "live"},
    )
    assert result["GHOST"]["momentum"]["source"] == "live"
    assert result["GHOST"]["reversal"]["source"] == "live"


def test_volatility_stays_pit_sourced_with_a_full_year_of_history():
    closes = [100.0 + (i % 5) for i in range(VOLATILITY_TRADING_DAYS)]
    result = _volatility_from_rows(
        ["AAPL"], {"AAPL": _pit_rows("AAPL", closes)},
        live_volatility_fn=lambda t: (_ for _ in ()).throw(AssertionError("should not hit live fallback")),
    )
    assert result["AAPL"]["source"] == "pit"
    assert result["AAPL"]["raw"] is not None


def test_volatility_falls_back_to_live_under_a_year_of_history():
    closes = [100.0 + i for i in range(37)]  # matches this session's real ~37-day PIT depth
    result = _volatility_from_rows(
        ["AAPL"], {"AAPL": _pit_rows("AAPL", closes)},
        live_volatility_fn=lambda t: {"raw": 22.5, "source": "live"},
    )
    assert result["AAPL"] == {"raw": 22.5, "source": "live"}


def test_compute_rsi_matches_technical_service_formula_on_a_rising_series():
    # A strictly rising series has zero losses -> RSI should be 100.
    closes = [100.0 + i for i in range(20)]
    assert _compute_rsi(closes) == 100.0


def test_compute_rsi_none_with_too_few_rows():
    assert _compute_rsi([100.0, 101.0]) is None


def test_annualized_volatility_zero_for_constant_prices():
    vol = _annualized_volatility(pd.Series([100.0] * 30))
    assert vol == 0.0


def test_annualized_volatility_none_for_single_price():
    assert _annualized_volatility(pd.Series([100.0])) is None


def test_blend_growth_averages_both_when_present():
    assert blend_growth(10.0, 20.0) == 15.0


def test_blend_growth_uses_whichever_one_is_present():
    assert blend_growth(None, 20.0) == 20.0
    assert blend_growth(10.0, None) == 10.0


def test_blend_growth_none_when_neither_present():
    assert blend_growth(None, None) is None


def test_blend_quality_averages_both_when_present():
    assert blend_quality(20.0, 10.0) == 15.0


def test_blend_quality_uses_whichever_one_is_present():
    assert blend_quality(None, 10.0) == 10.0
    assert blend_quality(20.0, None) == 20.0


def test_blend_quality_none_when_neither_present():
    assert blend_quality(None, None) is None


def test_latest_eps_surprise_picks_most_recent_reported_quarter():
    df = pd.DataFrame(
        {"Reported EPS": [1.5, 1.8, None], "Surprise(%)": [3.1, 6.7, None]},
        index=pd.to_datetime(["2025-10-30", "2026-01-29", "2026-04-30"]),
    )
    assert _latest_eps_surprise(df) == 6.7


def test_latest_eps_surprise_skips_upcoming_unreported_rows():
    df = pd.DataFrame(
        {"Reported EPS": [None], "Surprise(%)": [None]},
        index=pd.to_datetime(["2026-07-30"]),
    )
    assert _latest_eps_surprise(df) is None


def test_latest_eps_surprise_none_when_surprise_column_missing():
    df = pd.DataFrame({"Reported EPS": [1.5]}, index=pd.to_datetime(["2026-01-29"]))
    assert _latest_eps_surprise(df) is None


def test_latest_eps_surprise_none_for_empty_frame():
    assert _latest_eps_surprise(pd.DataFrame()) is None


def test_eps_revision_from_trend_computes_30day_pct_change():
    df = pd.DataFrame({"current": [2.02], "30daysAgo": [2.00]}, index=["0q"])
    # (2.02 - 2.00) / abs(2.00) * 100 = 1.0
    assert _eps_revision_from_trend(df) == 1.0


def test_eps_revision_from_trend_none_when_0q_row_missing():
    df = pd.DataFrame({"current": [1.0], "30daysAgo": [1.0]}, index=["+1q"])
    assert _eps_revision_from_trend(df) is None


def test_eps_revision_from_trend_none_when_30days_ago_is_zero():
    df = pd.DataFrame({"current": [1.0], "30daysAgo": [0.0]}, index=["0q"])
    assert _eps_revision_from_trend(df) is None


def test_eps_revision_from_trend_none_for_empty_frame():
    assert _eps_revision_from_trend(pd.DataFrame()) is None
