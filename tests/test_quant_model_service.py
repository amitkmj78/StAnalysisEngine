from datetime import date

import pandas as pd

from services.quant_model_service import as_calendar_dates, daily_model_returns, equity_snapshots_from_returns

IDX = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"])


def test_equal_weight_mean_of_picks_with_no_cost():
    closes = {"A": pd.Series([100.0, 110.0, 121.0], index=IDX), "B": pd.Series([50.0, 45.0, 40.5], index=IDX)}
    picks = {date(2026, 10, 1): ["A", "B"], date(2026, 10, 2): ["A"], date(2026, 10, 5): ["A"]}
    out = daily_model_returns(picks, closes, cost_bps_one_way=0)
    assert out == [(date(2026, 10, 2), 0.0), (date(2026, 10, 5), 10.0)]


def test_rebalance_cost_is_charged_per_period():
    closes = {"A": pd.Series([100.0, 110.0], index=IDX[:2])}
    picks = {date(2026, 10, 1): ["A"], date(2026, 10, 2): ["A"]}
    out = daily_model_returns(picks, closes, cost_bps_one_way=10)
    assert out[0][1] == 9.78  # 1.10 * (1 - 0.002) - 1: the 20 bps round trip comes off the compounded value


def test_missing_price_drops_that_pick_only():
    closes = {"A": pd.Series([100.0, 110.0], index=IDX[:2])}
    picks = {date(2026, 10, 1): ["A", "NOPE"], date(2026, 10, 2): ["A"]}
    out = daily_model_returns(picks, closes, cost_bps_one_way=0)
    assert out == [(date(2026, 10, 2), 10.0)]


def test_day_with_nothing_priced_is_skipped_not_guessed():
    closes = {"A": pd.Series([100.0, 110.0], index=IDX[:2])}
    picks = {date(2026, 10, 1): ["NOPE"], date(2026, 10, 2): ["A"], date(2026, 10, 5): ["A"]}
    out = daily_model_returns(picks, closes, cost_bps_one_way=0)
    assert all(d != date(2026, 10, 2) for d, _ in out)


def test_equity_chain_compounds_from_rebase_point():
    snaps = equity_snapshots_from_returns([(date(2026, 10, 2), 10.0), (date(2026, 10, 5), -10.0)])
    assert [round(s["equity"], 2) for s in snaps] == [11000.0, 9900.0]


def test_tz_aware_price_index_matches_plain_publication_dates():
    tz_index = pd.DatetimeIndex(["2026-10-01", "2026-10-02"]).tz_localize("America/New_York")
    closes = {"A": as_calendar_dates(pd.Series([100.0, 110.0], index=tz_index))}
    picks = {date(2026, 10, 1): ["A"], date(2026, 10, 2): ["A"]}
    assert daily_model_returns(picks, closes, cost_bps_one_way=0) == [(date(2026, 10, 2), 10.0)]
