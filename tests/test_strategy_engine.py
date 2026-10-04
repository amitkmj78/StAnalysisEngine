import numpy as np
import pandas as pd
import pytest

from services.strategy_engine import (
    ProtectiveExits,
    Rule,
    basket_returns,
    feature_frame,
    metrics,
    rule_mask,
    run_backtest,
    run_strategy,
    state_warnings,
)


def _prices(close, open_=None, high=None, low=None):
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    close = pd.Series(close, index=idx, dtype=float)
    open_ = close if open_ is None else pd.Series(open_, index=idx, dtype=float)
    high = close if high is None else pd.Series(high, index=idx, dtype=float)
    low = close if low is None else pd.Series(low, index=idx, dtype=float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})


def _frame_with_rsi(rsi, opens=None, closes=None, highs=None, lows=None):
    n = len(rsi)
    closes = [100.0] * n if closes is None else closes
    prices = _prices(closes, opens, highs, lows)
    frame = feature_frame(prices)
    frame["rsi_14"] = rsi
    return frame


def _trend_prices(n=420, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return _prices(close)


def test_rule_parse_rejects_unknown_fields_and_bad_regime_labels():
    with pytest.raises(ValueError, match="unknown field"):
        Rule.parse({"field": "eps", "op": ">", "value": 1})
    with pytest.raises(ValueError, match="regime labels"):
        Rule.parse({"field": "regime", "op": "is", "value": "Sunny"})
    with pytest.raises(ValueError, match="needs a number"):
        Rule.parse({"field": "rsi_14", "op": ">", "value": "high"})


def test_crosses_above_fires_only_on_the_bar_where_the_line_is_crossed():
    frame = pd.DataFrame({"rsi_14": [40.0, 45.0, 55.0, 60.0]}, index=pd.bdate_range("2021-01-01", periods=4))
    mask = rule_mask(frame, Rule("rsi_14", "crosses_above", 50.0))
    assert list(mask) == [False, False, True, False]


def test_cooldown_stops_a_re_entry_right_after_an_exit():
    # Crossing above 50 at close 1 buys at open 2; RSI 65 at close 3 sells at open 4.
    # Crossing again at close 5 would re-buy at once, but a 5-session cooldown holds it off.
    rsi = [40, 55, 55, 65, 40, 55, 55, 55, 65, 40, 40, 40]
    frame = _frame_with_rsi(rsi)
    entry = [Rule("rsi_14", "crosses_above", 50.0)]
    exit_ = [Rule("rsi_14", ">", 60.0)]
    _, _, with_cooldown = run_strategy({"AAA": frame}, entry, exit_, ProtectiveExits(), cooldown=5,
                                       cost_bps=0, slippage_bps=0)
    _, _, no_cooldown = run_strategy({"AAA": frame}, entry, exit_, ProtectiveExits(), cooldown=0,
                                     cost_bps=0, slippage_bps=0)
    assert len(with_cooldown["AAA"].trades) == 1
    assert len(no_cooldown["AAA"].trades) == 2


def test_a_gap_down_through_the_stop_fills_at_the_open_not_the_stop():
    close = [100, 100, 90, 90, 90]
    opens = [100, 100, 90, 90, 90]
    lows = [100, 100, 88, 90, 90]
    frame = _frame_with_rsi([10, 50, 50, 50, 50], opens=opens, closes=close, lows=lows)
    frames = {"AAA": frame}
    _, _, runs = run_strategy(frames, [Rule("rsi_14", "<", 20.0)], [], ProtectiveExits(stop_loss_pct=5),
                              cooldown=0, cost_bps=0, slippage_bps=0)
    trade = runs["AAA"].trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(90.0)  # the open, which is below the 95 stop
    assert trade.return_pct == pytest.approx(-10.0)


def test_an_intraday_stop_fills_at_the_stop_price():
    close = [100, 100, 97, 97, 97]
    opens = [100, 100, 100, 97, 97]
    lows = [100, 99, 94, 96, 97]
    frame = _frame_with_rsi([10, 50, 50, 50, 50], opens=opens, closes=close, lows=lows)
    _, _, runs = run_strategy({"AAA": frame}, [Rule("rsi_14", "<", 20.0)], [], ProtectiveExits(stop_loss_pct=5),
                              cooldown=0, cost_bps=0, slippage_bps=0)
    trade = runs["AAA"].trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(95.0)


def test_protective_exit_is_required_unless_waived():
    frames = {"AAA": feature_frame(_trend_prices())}
    bench = _trend_prices()["Close"]
    entry = [{"field": "rsi_14", "op": "<", "value": 40}]
    with pytest.raises(ValueError, match="protective exit"):
        run_backtest(frames, entry, [], bench)
    result = run_backtest(frames, entry, [], bench, waive_protective_exit=True)
    assert result["protective_exit"]["waived"] is True
    assert any(c["label"] == "Protective exit" for c in result["checks"])


def test_an_indicator_exit_counts_as_protective():
    frames = {"AAA": feature_frame(_trend_prices())}
    bench = _trend_prices()["Close"]
    result = run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 40}],
                          [{"field": "rsi_14", "op": ">", "value": 60}], bench)
    assert result["protective_exit"]["waived"] is False


def test_the_same_basket_with_no_costs_matches_the_stock_moves():
    prices = _trend_prices(n=300)
    frames = {"AAA": feature_frame(prices), "BBB": feature_frame(prices)}
    b = basket_returns(frames, cost_bps=0, slippage_bps=0)
    expected = prices["Close"].pct_change().fillna(0) * 100
    assert np.allclose(b.to_numpy(), expected.to_numpy(), atol=1e-9)


def test_costs_reduce_cagr_and_are_reported_as_drag():
    frames = {"AAA": feature_frame(_trend_prices(seed=4))}
    bench = _trend_prices(seed=5)["Close"]
    result = run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 45}],
                          [{"field": "rsi_14", "op": ">", "value": 55}], bench,
                          exits_raw={"stop_loss_pct": 10})
    if result["trades"]:
        assert result["cost_drag"]["cagr_points"] > 0
    assert result["cost_drag"]["total_costs_pct_of_equity"] >= 0


def test_a_rule_that_never_holds_reports_no_trades_with_a_warning():
    frames = {"AAA": feature_frame(_trend_prices())}
    bench = _trend_prices()["Close"]
    result = run_backtest(
        frames,
        [{"field": "close_vs_sma_50_pct", "op": ">", "value": 500}],
        [{"field": "rsi_14", "op": ">", "value": 60}],
        bench,
    )
    assert result["trades"] == 0
    assert any(c["label"] == "Enough trades" and c["status"] == "caution" for c in result["checks"])


def test_state_entry_with_an_exit_at_the_same_level_is_flagged():
    warnings = state_warnings([Rule("rsi_14", ">", 50.0)], [Rule("rsi_14", ">", 60.0)])
    assert warnings and "also meets the entry condition" in warnings[0]


def test_metrics_for_a_constant_daily_gain_match_the_compounding():
    idx = pd.bdate_range("2021-01-01", periods=252)
    m = metrics(pd.Series([1.0] * 252, index=idx))
    assert m["total_return_pct"] == pytest.approx((1.01 ** 252 - 1) * 100, rel=1e-4)
    assert m["cagr_pct"] == pytest.approx(m["total_return_pct"], rel=1e-4)


def test_many_variants_triggers_the_warning_and_both_periods_are_reported():
    frames = {"AAA": feature_frame(_trend_prices(seed=1)), "BBB": feature_frame(_trend_prices(seed=2))}
    bench = _trend_prices(seed=3)["Close"]
    result = run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 45}],
                          [{"field": "rsi_14", "op": ">", "value": 60}], bench, variants_tried=12)
    assert result["chance_sharpe_bar"] > 0
    assert result["in_sample"]["days"] > result["out_of_sample"]["days"] > 0
    assert result["benchmark_spy"]["days"] > 0


def test_regime_rules_use_the_stored_label_for_each_date():
    prices = _trend_prices(n=300)
    labels = {ts.strftime("%Y-%m-%d"): ("Risk-On" if i < 150 else "Cautious") for i, ts in enumerate(prices.index)}
    frame = feature_frame(prices, regime_by_date=labels)
    assert frame["regime"].iloc[0] == "Risk-On" and frame["regime"].iloc[-1] == "Cautious"
