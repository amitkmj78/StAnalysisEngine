from services.goal_links import sleeve_drift, steering, target_mix


def test_a_long_horizon_allows_more_stock_and_the_picks_cap_follows_the_horizon():
    assert target_mix(20)["stock_pct"] == 90.0
    assert target_mix(5)["stock_pct"] == 60.0
    assert target_mix(20)["picks_cap_pct"] == 20.0 and target_mix(5)["picks_cap_pct"] == 10.0
    mix = target_mix(12)
    assert abs(mix["stock_pct"] - (mix["core_pct"] + mix["picks_pct"])) < 1e-6


def test_drift_flags_a_role_more_than_five_points_off_target():
    targets = target_mix(20)
    holdings = [{"ticker": "VTI", "role": "core", "value": 500.0}, {"ticker": "NVDA", "role": "pick", "value": 500.0}]
    drift = sleeve_drift(holdings, targets)
    assert drift["flags"]
    assert any("Stock picks" in f for f in drift["flags"])


def test_steering_buys_with_new_money_and_reports_leftover_cash():
    targets = target_mix(20)
    holdings = [{"ticker": "VTI", "role": "core", "value": 700.0}, {"ticker": "NVDA", "role": "pick", "value": 100.0}]
    plan = steering(1500, holdings, targets, {"VTI": 100.0, "NVDA": 50.0})
    spent = sum(b["spent"] for b in plan["buys"])
    assert abs(spent + plan["cash"] - plan["into_stocks"]) < 0.01
    assert all(b["amount"] >= 0 for b in plan["buys"])
    assert plan["toward_bonds_outside_portfolio"] == round(1500 - plan["into_stocks"], 2)
