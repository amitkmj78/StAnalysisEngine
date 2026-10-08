from services.community_leaderboard_service import build_leaderboard
from services.quant_model_service import MODEL_MEMBER_LABEL


def _row(author="u1", display_name="Alice", excess=1.0, outcome="hit", direction="LONG", is_model=False):
    return {
        "author_user_id": None if is_model else author,
        "is_model": is_model,
        "display_name": display_name,
        "direction": direction,
        "realized_return_pct": excess,
        "excess_vs_spy_pct": excess,
        "outcome": outcome,
    }


def test_groups_by_author_and_computes_aggregates():
    rows = [_row(author="u1", excess=2.0), _row(author="u1", excess=-1.0, outcome="miss")]
    board = build_leaderboard(rows, min_samples=2)
    assert len(board) == 1
    entry = board[0]
    assert entry["num_ideas"] == 2
    assert entry["avg_excess_vs_spy_pct"] == 0.5
    assert entry["hit_rate_pct"] == 50.0
    assert entry["score"] is not None


def test_author_below_min_sample_still_listed_but_unscored():
    rows = [_row(author="u1", excess=5.0)]
    board = build_leaderboard(rows, min_samples=5)
    assert len(board) == 1
    assert board[0]["score"] is None
    assert board[0]["num_ideas"] == 1


def test_unscored_authors_always_sort_last():
    scored = [_row(author="good", excess=v) for v in [2.0, -1.0, 3.0, 0.5, 1.0]]
    unscored = [_row(author="thin", excess=5.0)]
    board = build_leaderboard(scored + unscored, min_samples=5)
    assert board[0]["author_user_id"] == "good"
    assert board[-1]["author_user_id"] == "thin"
    assert board[-1]["score"] is None


def test_model_author_appears_with_the_shared_sentinel_label():
    rows = [_row(is_model=True, excess=v) for v in [1.0, 2.0, 0.5, -0.5, 1.5]]
    board = build_leaderboard(rows, min_samples=5)
    assert board[0]["is_model"] is True
    assert board[0]["author_user_id"] is None
    assert board[0]["display_name"] == MODEL_MEMBER_LABEL


def test_worst_idea_included_per_author():
    rows = [
        _row(author="u1", excess=-10.0, direction="LONG", outcome="miss"),
        _row(author="u1", excess=3.0, direction="LONG", outcome="hit"),
    ]
    board = build_leaderboard(rows, min_samples=2)
    assert board[0]["worst_idea"]["realized_return_pct"] == -10.0


def test_higher_risk_adjusted_score_ranks_first():
    # Same mean (1.0) as a near-zero-variance set would have, but with
    # enough spread to avoid the "identical values -> undefined, not
    # infinite" None case risk_adjusted_excess_return deliberately
    # returns for truly zero variance.
    steady = [_row(author="steady", excess=v) for v in [1.1, 0.9, 1.0, 1.05, 0.95]]
    volatile = [_row(author="volatile", excess=v) for v in [10.0, -8.0, 5.0, -3.0, 1.0]]
    board = build_leaderboard(steady + volatile, min_samples=5)
    assert board[0]["author_user_id"] == "steady"
