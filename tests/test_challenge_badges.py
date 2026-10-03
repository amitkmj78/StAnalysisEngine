from services.challenge_leaderboard import compute_badges


def _e(user, score, dd, vol, days=10, model=False):
    return {"user_id": user, "score": score, "max_drawdown_pct": dd,
            "annualized_volatility_pct": vol, "days_of_data": days, "is_model": model}


def _names(badges, i):
    return [b["badge"] for b in badges[i]]


def test_top_three_by_score_get_places():
    entries = [_e("a", 5, -1, 10), _e("b", 4, -2, 12), _e("c", 3, -3, 14), _e("d", 2, -4, 16)]
    out = compute_badges(entries, ended=True)
    assert _names(out, 0) == ["Top 1", "Lowest Drawdown", "Most Consistent"]
    assert "Top 3" in _names(out, 2)
    assert "Top" not in " ".join(_names(out, 3))


def test_lowest_drawdown_and_most_consistent_use_risk_history_only():
    entries = [_e("a", 1, -2, 20), _e("b", 2, -9, 5, days=3)]  # b has too little history for risk badges
    out = compute_badges(entries, ended=False)
    assert "Lowest Drawdown" in _names(out, 0)
    assert "Most Consistent" in _names(out, 0)
    assert "Most Consistent" not in _names(out, 1)


def test_model_never_earns_a_badge():
    entries = [_e("a", 1, -5, 15), _e(None, 9, -0.5, 4, model=True)]
    out = compute_badges(entries, ended=True)
    assert out[1] == []  # the model scores highest but earns nothing
    assert "Top 1" in _names(out, 0)


def test_labels_say_current_while_running_and_final_once_ended():
    running = compute_badges([_e("a", 1, -2, 10)], ended=False)
    final = compute_badges([_e("a", 1, -2, 10)], ended=True)
    assert "current" in running[0][0]["detail"] and "final" in final[0][0]["detail"]
