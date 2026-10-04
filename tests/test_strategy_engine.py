import numpy as np
import pandas as pd
import pytest

from services.strategy_engine import (
    Rule,
    feature_frame,
    metrics,
    run_backtest,
    simulate,
    ticker_position,
)


def _prices(close, open_=None):
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    close = pd.Series(close, index=idx, dtype=float)
    open_ = close if open_ is None else pd.Series(open_, index=idx, dtype=float)
    return pd.DataFrame({"Open": open_, "Close": close})


def test_rule_parse_rejects_unknown_fields_and_bad_regime_labels():
    with pytest.raises(ValueError, match="unknown field"):
        Rule.parse({"field": "eps", "op": ">", "value": 1})
    with pytest.raises(ValueError, match="regime labels"):
        Rule.parse({"field": "regime", "op": "is", "value": "Sunny"})
    with pytest.raises(ValueError, match="needs a number"):
        Rule.parse({"field": "rsi_14", "op": ">", "value": "high"})
    assert Rule.parse({"field": "rsi_14", "op": "<", "value": "30"}).value == 30.0


def test_crosses_above_fires_only_on_the_bar_where_the_line_is_crossed():
    frame = pd.DataFrame({"rsi_14": [40.0, 45.0, 55.0, 60.0]}, index=pd.bdate_range("2021-01-01", periods=4))
    from services.strategy_engine import rule_mask
    mask = rule_mask(frame, Rule("rsi_14", "crosses_above", 50.0))
    assert list(mask) == [False, False, True, False]


def test_a_signal_at_one_close_fills_at_the_next_open():
    close = [100, 100, 100, 100, 100, 100]
    frame = feature_frame(_prices(close, open_=[100, 100, 100, 110, 110, 110]))
    frame["rsi_14"] = [10, 40, 40, 40, 40, 40]  # entry rule true at day 0 only
    pos = ticker_position(frame, [Rule("rsi_14", "<", 20.0)], [Rule("rsi_14", ">", 90.0)])
    # rule read at day 0's close, so the buy fills at day 1's open and the position is held from day 1
    assert pos["trade"].iloc[1] == 1
    assert pos["held"].iloc[1] == 1
    assert pos["trade"].iloc[0] == 0


def test_flat_prices_with_one_round_trip_lose_only_costs():
    close = [100.0] * 30
    frame = feature_frame(_prices(close))
    frame["rsi_14"] = [10.0] + [50.0] * 9 + [95.0] + [50.0] * 19  # entry on day 0, exit on day 10
    daily = simulate({"AAA": frame}, [Rule("rsi_14", "<", 20.0)], [Rule("rsi_14", ">", 90.0)])
    # 10 bps cost + 5 bps slippage on each side = 15 bps; two sides = 0.30%
    assert daily.sum() == pytest.approx(-0.30, abs=0.01)


def test_metrics_for_a_constant_daily_gain_match_the_compounding():
    idx = pd.bdate_range("2021-01-01", periods=252)
    m = metrics(pd.Series([1.0] * 252, index=idx))
    assert m["total_return_pct"] == pytest.approx((1.01 ** 252 - 1) * 100, rel=1e-4)
    assert m["cagr_pct"] == pytest.approx(m["total_return_pct"], rel=1e-4)  # exactly one year
    assert m["max_drawdown_pct"] == pytest.approx(0.0)


def _trend_prices(n=420, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return _prices(close)


def test_many_variants_triggers_the_overfitting_warning_and_the_result_has_both_periods():
    frames = {"AAA": feature_frame(_trend_prices(seed=1)), "BBB": feature_frame(_trend_prices(seed=2))}
    bench = _trend_prices(seed=3)["Close"]
    entry = [{"field": "rsi_14", "op": "<", "value": 45}]
    exit_ = [{"field": "rsi_14", "op": ">", "value": 60}]
    result = run_backtest(frames, entry, exit_, bench, variants_tried=12)
    assert any("12 rule variants" in w for w in result["warnings"])
    assert result["in_sample"]["days"] > result["out_of_sample"]["days"] > 0
    assert result["benchmark_spy"]["days"] > 0
    assert result["trades"] >= 0


def test_backtest_needs_both_entry_and_exit_rules_and_a_valid_ticker_count():
    frames = {"AAA": feature_frame(_trend_prices())}
    bench = _trend_prices()["Close"]
    with pytest.raises(ValueError, match="entry rule"):
        run_backtest(frames, [], [{"field": "rsi_14", "op": ">", "value": 60}], bench)
    with pytest.raises(ValueError, match="exit rule"):
        run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 40}], [], bench)


def test_regime_rules_use_the_stored_label_for_each_date():
    prices = _trend_prices(n=300)
    labels = {ts.strftime("%Y-%m-%d"): ("Risk-On" if i < 150 else "Cautious") for i, ts in enumerate(prices.index)}
    frame = feature_frame(prices, regime_by_date=labels)
    assert frame["regime"].iloc[0] == "Risk-On" and frame["regime"].iloc[-1] == "Cautious"


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
    assert any("No trades" in w for w in result["warnings"])
