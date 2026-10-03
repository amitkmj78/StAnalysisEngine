from services.challenge_notifications import passed_by, rank_by_user


def test_rank_by_user_ranks_only_scored_entries_in_score_order():
    entries = [
        {"user_id": "a", "score": 1.0},
        {"user_id": "b", "score": 3.0},
        {"user_id": "c", "score": None},
        {"user_id": None, "score": 9.0},  # the model: no user id, so not notified
    ]
    # The model (score 9.0) is ranked too, so it takes place 1, matching the page.
    assert rank_by_user(entries) == {"b": 2, "a": 3}


def test_passed_by_names_the_member_who_moved_above():
    prev = {"a": 1, "b": 2, "c": 3}
    cur = {"c": 1, "a": 2, "b": 3}  # c overtook both a and b
    labels = {"c": "ca***@gmail.com", "a": "aa***@gmail.com", "b": "bb***@gmail.com"}
    out = passed_by(prev, cur, labels)
    assert out == {"a": "ca***@gmail.com", "b": "ca***@gmail.com"}


def test_no_passed_message_when_rank_held_or_improved():
    prev = {"a": 2, "b": 1}
    cur = {"a": 1, "b": 2}  # a improved, b dropped because a moved above -- b is passed
    out = passed_by(prev, cur, {"a": "aa***@x", "b": "bb***@x"})
    assert "a" not in out and out["b"] == "aa***@x"
