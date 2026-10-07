from services.goal_links import long_term_alerts, sleeve_drift, steering, target_mix


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


def test_a_pick_above_its_cap_raises_a_plain_alert_and_small_moves_do_not():
    targets = target_mix(20)  # picks cap 20%
    holdings = [{"ticker": "NVDA", "role": "pick", "value": 600.0}, {"ticker": "VTI", "role": "core", "value": 400.0}]
    drift = sleeve_drift(holdings, targets)
    alerts = long_term_alerts(holdings, targets, drift, months_elapsed=3)
    assert any(a["kind"] == "over_cap" and a["ticker"] == "NVDA" for a in alerts)
    quiet = long_term_alerts([{"ticker": "VTI", "role": "core", "value": 1000.0}], targets, {"flags": []}, months_elapsed=3)
    assert quiet == []


def test_the_yearly_review_is_a_reminder_on_each_anniversary():
    targets = target_mix(20)
    assert any(a["kind"] == "yearly_review" for a in long_term_alerts([], targets, {"flags": []}, months_elapsed=12))
    assert not any(a["kind"] == "yearly_review" for a in long_term_alerts([], targets, {"flags": []}, months_elapsed=5))
