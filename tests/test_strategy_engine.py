import numpy as np
import pandas as pd
import pytest

from services.strategy_engine import (
    ProtectiveExits,
    Rule,
    basket_returns,
    evaluate_protective_exit,
    feature_frame,
    metrics,
    rule_mask,
    run_backtest,
    run_strategy,
    state_warnings,
    today_signal,
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


def test_a_full_weight_on_one_stock_matches_that_stock_alone():
    a = _trend_prices(n=300, seed=8)
    b = _trend_prices(n=300, seed=9)
    frames = {"AAA": feature_frame(a), "BBB": feature_frame(b)}
    only_a = basket_returns(frames, cost_bps=0, slippage_bps=0, weights={"AAA": 1.0, "BBB": 0.0})
    expected = a["Close"].pct_change().fillna(0) * 100
    assert np.allclose(only_a.to_numpy(), expected.to_numpy(), atol=1e-9)


def test_weights_must_match_the_tickers_tested():
    frames = {"AAA": feature_frame(_trend_prices(seed=1)), "BBB": feature_frame(_trend_prices(seed=2))}
    bench = _trend_prices(seed=3)["Close"]
    with pytest.raises(ValueError, match="exactly the tickers"):
        run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 45}],
                     [{"field": "rsi_14", "op": ">", "value": 55}], bench, weights={"AAA": 1.0})


def test_model_portfolio_is_withheld_until_twelve_months_of_history():
    from services.strategy_engine import model_portfolio_summary

    short = [["2026-01-01", 10000.0], ["2026-03-01", 10100.0], ["2026-05-01", 10200.0]]
    result = model_portfolio_summary(short)
    assert result["available"] is False and result["months_of_history"] < 12


def test_model_portfolio_reports_return_and_drawdown_once_it_has_a_year():
    from services.strategy_engine import model_portfolio_summary

    series = [["2025-01-01", 10000.0], ["2025-07-01", 12000.0], ["2025-12-31", 9000.0], ["2026-06-30", 13000.0]]
    result = model_portfolio_summary(series)
    assert result["available"] is True
    assert result["total_return_pct"] == 30.0
    assert result["max_drawdown_pct"] == -25.0


def test_turnover_counts_buys_and_sells_once_each():
    import numpy as np
    from services.strategy_engine import TickerRun, strategy_turnover_pct_per_year, PERIODS_PER_YEAR

    held = np.zeros(PERIODS_PER_YEAR)
    held[10:20] = 1  # one entry and one exit in a year: buys + sells = 2 sleeves, one way = 1 sleeve
    run = TickerRun(returns=np.zeros(len(held)), costs=np.zeros(len(held)), held=held, trades=[], exit_events=1, churn_events=0)
    assert strategy_turnover_pct_per_year({"AAA": run}, {"AAA": 1.0}, PERIODS_PER_YEAR) == 100.0


def test_the_backtest_reports_turnover_for_the_strategy_basket_and_spy():
    frames = {"AAA": feature_frame(_trend_prices(seed=7)), "BBB": feature_frame(_trend_prices(seed=8))}
    bench = _trend_prices(seed=9)["Close"]
    result = run_backtest(frames, [{"field": "rsi_14", "op": "<", "value": 45}], [], bench, waive_protective_exit=True)
    assert result["strategy"]["turnover_pct_per_year"] is not None
    assert result["basket"]["turnover_pct_per_year"] is not None
    assert result["benchmark_spy"]["turnover_pct_per_year"] == 0.0


def test_sessions_since_earnings_only_uses_reports_already_made():
    import pandas as pd
    from services.strategy_engine import sessions_since_reports

    index = pd.bdate_range("2026-01-01", periods=20)
    report = pd.Timestamp(index[10])  # reported on day 10
    values = sessions_since_reports(index, [report])
    assert np.isnan(values[10])  # the report day itself does not count yet
    assert values[11] == 0 and values[15] == 4
    later = sessions_since_reports(index, [report, pd.Timestamp(index[15])])
    assert np.allclose(later[:15], values[:15], equal_nan=True)  # a later report changes nothing before it
    assert later[16] == 0


def test_a_stock_cannot_be_bought_while_it_is_not_an_index_member():
    import pandas as pd
    from services.strategy_engine import Rule, _run_ticker, ProtectiveExits, feature_frame

    prices = _trend_prices(seed=11)
    flags = pd.Series(False, index=prices.index)
    flags.iloc[len(flags) // 2:] = True
    member_frame = feature_frame(prices, member_flags=flags)
    assert not member_frame["is_member"].iloc[: len(flags) // 2].any()
    entry = [Rule.parse({"field": "rsi_14", "op": "<", "value": 99})]  # true almost every day
    run = _run_ticker(member_frame, entry, [], ProtectiveExits(stop_loss_pct=None), 0, 0.0, "AAA")
    first_held = int(np.argmax(run.held > 0))
    assert first_held >= len(flags) // 2


def test_a_zero_threshold_is_still_tested_by_the_sensitivity_check():
    from services.strategy_robustness import sensitivity

    seen = []

    def fake_run(params):
        seen.append(params["entry[0] close_vs_sma_200_pct >"])
        return 1.0 + 0.1 * params["entry[0] close_vs_sma_200_pct >"]

    result = sensitivity(fake_run, {"entry[0] close_vs_sma_200_pct >": 0.0})
    assert result["rows"][0]["swing"] and result["rows"][0]["swing"] > 0
    assert -1.0 in seen and 1.0 in seen


# ---------------------------------------------------------------- evaluate_protective_exit (STB-6)

def test_evaluate_protective_exit_stop_loss_triggers():
    result = evaluate_protective_exit(100.0, float("nan"), 100.0, low=85.0, high=100.0, open_=92.0,
                                      exits=ProtectiveExits(stop_loss_pct=10))
    assert result == ("stop_loss", 90.0)  # min(open_=92, level=90)


def test_evaluate_protective_exit_prefers_the_tighter_of_stop_loss_and_atr_stop():
    # stop_loss level = 90 (entry 100, 10%); atr_stop level = 95 (entry 100 - 1*5) -- atr_stop is tighter (higher).
    exits = ProtectiveExits(stop_loss_pct=10, atr_stop_k=1.0)
    result = evaluate_protective_exit(100.0, atr_entry=5.0, peak=100.0, low=93.0, high=100.0, open_=98.0, exits=exits)
    assert result == ("atr_stop", 95.0)


def test_evaluate_protective_exit_ignores_atr_stop_when_atr_entry_is_nan():
    exits = ProtectiveExits(stop_loss_pct=10, atr_stop_k=1.0)
    # atr_stop would be entry - 1*nan -> NaN, must be skipped; low=93 doesn't breach the 90 stop_loss level.
    result = evaluate_protective_exit(100.0, atr_entry=float("nan"), peak=100.0, low=93.0, high=100.0, open_=98.0, exits=exits)
    assert result is None


def test_evaluate_protective_exit_trailing_stop_triggers():
    result = evaluate_protective_exit(100.0, float("nan"), peak=120.0, low=105.0, high=115.0, open_=110.0,
                                      exits=ProtectiveExits(trailing_stop_pct=10))
    assert result == ("trailing_stop", 108.0)


def test_evaluate_protective_exit_take_profit_triggers():
    result = evaluate_protective_exit(100.0, float("nan"), peak=100.0, low=100.0, high=125.0, open_=105.0,
                                      exits=ProtectiveExits(take_profit_pct=20))
    assert result == ("take_profit", 120.0)  # max(open_=105, target=120)


def test_evaluate_protective_exit_returns_none_when_nothing_fires():
    result = evaluate_protective_exit(100.0, float("nan"), peak=100.0, low=99.0, high=101.0, open_=100.0,
                                      exits=ProtectiveExits(stop_loss_pct=10, take_profit_pct=20))
    assert result is None


# ---------------------------------------------------------------- today_signal (STB-6)

def _last_row_frame(rsi_values, opens=None, highs=None, lows=None, closes=None):
    """A short feature frame whose LAST row is the one today_signal reads."""
    return _frame_with_rsi(rsi_values, opens=opens, closes=closes, highs=highs, lows=lows)


def test_today_signal_enters_when_flat_and_the_entry_rule_fires_on_the_last_row():
    frame = _last_row_frame([50, 50, 50, 25])  # last row's RSI is 25
    entry = [Rule.parse({"field": "rsi_14", "op": "<", "value": 30})]
    result = today_signal(frame, entry, [], ProtectiveExits(stop_loss_pct=10), held=False)
    assert result == {"action": "enter", "reason": "entry_rule"}


def test_today_signal_holds_when_flat_and_cooldown_is_active_even_if_entry_fires():
    frame = _last_row_frame([50, 50, 50, 25])
    entry = [Rule.parse({"field": "rsi_14", "op": "<", "value": 30})]
    result = today_signal(frame, entry, [], ProtectiveExits(stop_loss_pct=10), held=False, cooldown_active=True)
    assert result == {"action": "hold", "reason": "cooldown"}


def test_today_signal_holds_when_flat_and_the_entry_rule_does_not_fire():
    frame = _last_row_frame([50, 50, 50, 80])
    entry = [Rule.parse({"field": "rsi_14", "op": "<", "value": 30})]
    result = today_signal(frame, entry, [], ProtectiveExits(stop_loss_pct=10), held=False)
    assert result == {"action": "hold", "reason": None}


def test_today_signal_exits_on_a_protective_stop_breached_on_the_last_bar():
    frame = _last_row_frame([50, 50, 50, 50], opens=[100, 100, 100, 92], lows=[100, 100, 100, 85])
    result = today_signal(
        frame, [], [], ProtectiveExits(stop_loss_pct=10),
        held=True, entry_price=100.0, entry_atr=None, peak=100.0, days_held=3,
    )
    assert result == {"action": "exit", "reason": "stop_loss"}


def test_today_signal_skips_the_protective_check_on_the_entry_day_itself():
    # Mirrors _run_ticker's `i > entry_i` guard -- days_held=0 means this
    # position was just entered, so no protective check runs yet.
    frame = _last_row_frame([50, 50, 50, 50], opens=[100, 100, 100, 92], lows=[100, 100, 100, 85])
    result = today_signal(
        frame, [], [], ProtectiveExits(stop_loss_pct=10),
        held=True, entry_price=100.0, entry_atr=None, peak=100.0, days_held=0,
    )
    assert result == {"action": "hold", "reason": None}


def test_today_signal_exits_on_time_stop():
    frame = _last_row_frame([50, 50, 50, 50])
    result = today_signal(
        frame, [], [], ProtectiveExits(time_stop_sessions=3),
        held=True, entry_price=100.0, peak=100.0, days_held=3,
    )
    assert result == {"action": "exit", "reason": "time_stop"}


def test_today_signal_exits_on_the_indicator_exit_rule():
    frame = _last_row_frame([50, 50, 50, 80])
    exit_ = [Rule.parse({"field": "rsi_14", "op": ">", "value": 70})]
    result = today_signal(
        frame, [], exit_, ProtectiveExits(stop_loss_pct=90),  # stop far away, never fires
        held=True, entry_price=100.0, peak=100.0, days_held=3,
    )
    assert result == {"action": "exit", "reason": "indicator_exit"}


def test_today_signal_holds_an_open_position_when_nothing_fires():
    frame = _last_row_frame([50, 50, 50, 50])
    result = today_signal(
        frame, [], [], ProtectiveExits(stop_loss_pct=90),
        held=True, entry_price=100.0, peak=100.0, days_held=3,
    )
    assert result == {"action": "hold", "reason": None}
