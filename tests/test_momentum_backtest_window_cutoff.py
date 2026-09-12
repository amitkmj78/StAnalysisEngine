from unittest.mock import patch

import numpy as np
import pandas as pd

from services.momentum_backtest_service import _compute_backtest_momentum_ranking


def _price_series(start: str, n_days: int, base: float = 100.0) -> pd.Series:
    idx = pd.bdate_range(start=start, periods=n_days)
    rng = np.random.default_rng(seed=hash(start) % (2**32))
    # A gentle random walk so momentum ranking has something to differentiate.
    returns = rng.normal(0, 0.01, n_days)
    prices = base * np.cumprod(1 + returns)
    return pd.Series(prices, index=idx)


def _frame_for(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Close": series, "Volume": [1_000_000] * len(series)}, index=series.index)


def test_a_recently_listed_ticker_does_not_shrink_the_whole_window():
    """Regression test: a recent spinoff/IPO with only a few weeks of
    history was passing the per-ticker length filter, then poisoning the
    hard date INTERSECTION for the entire universe down to its own short
    history -- production hit this live (two tickers with 61/74 days of
    history collapsed a "3-year" backtest to about 2 months of data,
    still returning a plausible-looking 200 OK result). The recent
    ticker must be excluded instead, leaving the long-history tickers'
    shared multi-year window intact."""
    long_history_tickers = {
        f"LONG{i}": _frame_for(_price_series("2023-01-03", 750, base=100.0 + i))
        for i in range(6)  # top_n(2) + 1 required, plenty of margin
    }
    recent_ticker = {"RECENT": _frame_for(_price_series("2026-07-01", 45, base=50.0))}
    frames = {**long_history_tickers, **recent_ticker}

    with patch(
        "services.momentum_backtest_service._universe_tickers",
        return_value=list(frames.keys()),
    ), patch(
        "services.momentum_backtest_service._download_universe_history",
        return_value=frames,
    ):
        result = _compute_backtest_momentum_ranking(
            asset_type="Stock",
            universe_key="Test",
            lookback_days=10,
            top_n=2,
            years=2,
            horizon_days=10,
            slippage_bps=0.0,
            commission_bps=0.0,
            borrow_cost_bps_annual=0.0,
            risk_free_rate_annual=0.0,
        )

    assert result is not None
    # ~2 years of business days at a 10-day horizon should yield dozens
    # of periods, not the ~2-3 a collapsed-to-45-days window would give.
    assert result["num_periods"] > 20
    # RECENT never appears in any period's picks -- it was excluded
    # up front, not just unlikely to be top-ranked.
    all_picks = {t for p in result["periods"] for t in p["picks"]}
    assert "RECENT" not in all_picks


def test_all_tickers_too_recent_returns_none_not_a_crash():
    frames = {
        f"T{i}": _frame_for(_price_series("2026-07-01", 45, base=100.0 + i)) for i in range(6)
    }
    with patch(
        "services.momentum_backtest_service._universe_tickers", return_value=list(frames.keys())
    ), patch(
        "services.momentum_backtest_service._download_universe_history", return_value=frames
    ):
        result = _compute_backtest_momentum_ranking(
            asset_type="Stock",
            universe_key="Test",
            lookback_days=10,
            top_n=2,
            years=2,
            horizon_days=10,
            slippage_bps=0.0,
            commission_bps=0.0,
            borrow_cost_bps_annual=0.0,
            risk_free_rate_annual=0.0,
        )
    assert result is None
