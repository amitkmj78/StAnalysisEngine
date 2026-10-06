import pytest

from services.rule_text import parse_rule_text


def test_plain_english_lines_become_the_engine_rules():
    parsed = parse_rule_text("""
entry:
RSI(14) crosses above 30
Close above SMA(200)
exit:
RSI(14) above 70
stop loss 8%
wait 5 days after selling
""")
    assert parsed["entry"] == [
        {"field": "rsi_14", "op": "crosses_above", "value": 30.0},
        {"field": "close_vs_sma_200_pct", "op": ">", "value": 0.0},
    ]
    assert parsed["exit"] == [{"field": "rsi_14", "op": ">", "value": 70.0}]
    assert parsed["exits"]["stop_loss_pct"] == 8.0
    assert parsed["cooldown_sessions"] == 5


def test_a_line_that_is_not_understood_is_reported_with_its_number():
    with pytest.raises(ValueError, match="Line 2"):
        parse_rule_text("entry:\nbuy when the moon is full\n")


def test_a_rule_is_required():
    with pytest.raises(ValueError, match="at least one buy rule"):
        parse_rule_text("exit:\nstop loss 5%\n")


def test_the_two_moving_averages_and_volume_forms_parse():
    parsed = parse_rule_text("SMA(20) crosses above SMA(50)\nvolume vs 20-day average above 50%\nwithin 2% of 52-week high\n")
    assert parsed["entry"][0] == {"field": "sma_20_vs_50_pct", "op": "crosses_above", "value": 0.0}
    assert parsed["entry"][1] == {"field": "volume_vs_20d_pct", "op": ">", "value": 50.0}
    assert parsed["entry"][2] == {"field": "dist_52w_high_pct", "op": ">", "value": -2.0}


def test_the_sdk_runs_plain_english_rules_through_the_engine(monkeypatch):
    import numpy as np
    import pandas as pd
    import strategy_sdk

    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 800)))
    idx = pd.bdate_range("2019-01-01", periods=800)
    frame = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6}, index=idx)
    monkeypatch.setattr(strategy_sdk, "get_cached_history", lambda t, period, adj, interval: frame)
    result = strategy_sdk.backtest_text(["AAA", "BBB"], "entry:\nClose above SMA(200)\nexit:\nClose below SMA(200)\nstop loss 8%\n", explain=False)
    assert "strategy" in result
