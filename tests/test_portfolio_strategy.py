from unittest.mock import patch

import pandas as pd

from services.portfolio_strategy import (
    EnrichedPosition,
    _compute_long_term_plan,
    _compute_short_term_plan,
    _compute_short_term_targets,
    _long_term_technical_clause,
    _short_term_technical_clause,
    _ticker_technicals,
    _volatility_multiplier,
    build_robinhood_strategies,
)


def _pos(ticker="AAA", current_price=100.0, avg_cost=100.0, pnl_pct=-5.0, risk_profile="Balanced", risk_factor=5):
    return EnrichedPosition(
        ticker=ticker,
        shares=10,
        avg_cost=avg_cost,
        current_price=current_price,
        pnl_pct=pnl_pct,
        risk_profile=risk_profile,
        risk_factor=risk_factor,
    )


def _history(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=idx)


def test_ticker_technicals_returns_none_on_short_history():
    with patch("services.portfolio_strategy.get_cached_history", return_value=_history([100.0] * 5)):
        assert _ticker_technicals("THIN") is None


def test_ticker_technicals_returns_none_on_fetch_failure():
    with patch("services.portfolio_strategy.get_cached_history", side_effect=Exception("boom")):
        assert _ticker_technicals("BROKEN") is None


def test_volatility_multiplier_defaults_to_one_without_technicals():
    assert _volatility_multiplier(None) == 1.0
    assert _volatility_multiplier({"ann_vol_pct": None}) == 1.0


def test_two_positions_same_pnl_bucket_get_different_targets_and_stance_from_technicals():
    """The bug this is fixing: two tickers at the same risk profile/factor
    and the same PnL% bucket previously got byte-identical target/stop %
    and Stance text (see services/portfolio_strategy.py's PnL if/elif
    chain, which never looked at the ticker itself). A real per-ticker
    volatility/trend reading must now tell them apart."""
    calm_pos = _pos(ticker="CALM", pnl_pct=-5.0)
    wild_pos = _pos(ticker="WILD", pnl_pct=-5.0)

    # CALM: tiny day-to-day wiggle -> low realized volatility.
    calm_closes = [100.0 + (i % 2) * 0.1 for i in range(60)]
    # WILD: large alternating daily swings -> high realized volatility
    # (a smooth, constant-return compounding series would have ~zero
    # realized stdev despite the big cumulative move, so this needs
    # actual day-to-day variability, not just a trend).
    wild_closes = []
    price = 100.0
    for i in range(60):
        price *= 1.08 if i % 2 == 0 else 1 / 1.04
        wild_closes.append(price)

    with patch("services.portfolio_strategy.get_cached_history", return_value=_history(calm_closes)):
        calm_technicals = _ticker_technicals("CALM")
    with patch("services.portfolio_strategy.get_cached_history", return_value=_history(wild_closes)):
        wild_technicals = _ticker_technicals("WILD")

    calm_target, calm_stop = _compute_short_term_targets(calm_pos, calm_technicals)
    wild_target, wild_stop = _compute_short_term_targets(wild_pos, wild_technicals)

    # WILD's real volatility is far above CALM's -> a wider band, not the
    # identical +8.0%/-4.5% both would have gotten before this fix.
    assert (wild_target - wild_pos.current_price) > (calm_target - calm_pos.current_price)
    assert (wild_pos.current_price - wild_stop) > (calm_pos.current_price - calm_stop)

    calm_plan = _compute_short_term_plan(calm_pos, calm_technicals)
    wild_plan = _compute_short_term_plan(wild_pos, wild_technicals)
    assert calm_plan.split("**Stance:**")[1] != wild_plan.split("**Stance:**")[1]


def test_short_term_technical_clause_reflects_trend_direction():
    assert _short_term_technical_clause(None) is None
    assert _short_term_technical_clause({"month_trend_pct": None}) is None
    assert "positive" in _short_term_technical_clause({"month_trend_pct": 10.0})
    assert "sharply negative" in _short_term_technical_clause({"month_trend_pct": -10.0})
    assert "range-bound" in _short_term_technical_clause({"month_trend_pct": 0.5})


def test_long_term_technical_clause_reflects_52_week_range_position():
    assert _long_term_technical_clause(None) is None
    assert "52-week high" in _long_term_technical_clause({"range_position": 0.9})
    assert "52-week low" in _long_term_technical_clause({"range_position": 0.1})
    assert "52-week range" in _long_term_technical_clause({"range_position": 0.5})


def test_plans_still_work_when_technicals_unavailable():
    """Regression safety: a ticker with no usable history (new listing,
    Yahoo/Alpaca both failing) must fall back to exactly the prior
    risk-profile-only behavior, not raise or produce garbage text."""
    pos = _pos(ticker="NEWCO", pnl_pct=-5.0)
    short_plan = _compute_short_term_plan(pos, None)
    long_plan = _compute_long_term_plan(pos, None)
    target, stop = _compute_short_term_targets(pos, None)
    assert "Stance:" in short_plan
    assert "Risk framing:" in long_plan
    assert target > pos.current_price > stop


def test_build_robinhood_strategies_fetches_technicals_once_per_position():
    holdings_df = pd.DataFrame(
        [{"Ticker": "AAA", "Shares": 10, "Avg_Cost": 100.0, "Current_Price": 100.0}]
    )
    with patch("services.portfolio_strategy._ticker_technicals", return_value=None) as mock_technicals:
        strat_df = build_robinhood_strategies(holdings_df, risk_profile="Balanced", risk_factor=5)
    assert len(strat_df) == 1
    mock_technicals.assert_called_once_with("AAA")
