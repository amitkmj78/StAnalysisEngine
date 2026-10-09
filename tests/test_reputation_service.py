from services.reputation_service import compute_reputation


def _idea(excess, outcome="hit", author="u1", display_name="Author"):
    return {
        "author_user_id": author, "is_model": False, "direction": "LONG",
        "realized_return_pct": excess, "excess_vs_spy_pct": excess, "outcome": outcome,
        "display_name": display_name,
    }


def test_reputation_none_below_sample_gate():
    rows = [_idea(1.0) for _ in range(5)]  # below MIN_IDEAS_FOR_LEADERBOARD (10)

    result = compute_reputation(rows)

    assert result["track_record_component"] is None
    assert result["reputation"] is None
    assert result["num_ideas_scored"] == 5


def test_reputation_qa_component_is_always_zero_placeholder():
    """SOC-8: accepted answers (BEG-4) are not built this round -- the
    qa_component must never silently inflate the total."""
    result = compute_reputation([])

    assert result["qa_component"] == 0
    assert result["num_ideas_scored"] == 0
    assert result["reputation"] is None


def test_reputation_real_score_above_sample_gate():
    # Varying excess returns above the min-sample gate -- a real,
    # non-None risk-adjusted score should come through.
    excesses = [1.0, 2.0, -0.5, 1.5, 0.8, 1.2, -0.2, 2.1, 0.5, 1.8]
    rows = [_idea(e) for e in excesses]

    result = compute_reputation(rows)

    assert result["track_record_component"] is not None
    assert result["reputation"] == result["track_record_component"]
    assert result["num_ideas_scored"] == 10
