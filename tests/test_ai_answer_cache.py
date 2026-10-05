from services.ai_answer_cache import MAX_ENTRIES, TTL_SECONDS, ai_answer_key, clear_ai_answers, get_ai_answer, put_ai_answer


def setup_function():
    clear_ai_answers()


def test_the_same_question_about_the_same_stock_today_shares_one_key():
    a = ai_answer_key("cited-ticker", "AAPL", "How is Apple doing?", day="2026-10-05")
    b = ai_answer_key("cited-ticker", "aapl", "  how  is apple DOING? ", day="2026-10-05")
    assert a == b


def test_a_new_day_or_another_stock_gets_its_own_answer():
    base = ai_answer_key("cited-ticker", "AAPL", "How is Apple doing?", day="2026-10-05")
    assert base != ai_answer_key("cited-ticker", "AAPL", "How is Apple doing?", day="2026-10-06")
    assert base != ai_answer_key("cited-ticker", "MSFT", "How is Apple doing?", day="2026-10-05")


def test_a_stored_answer_comes_back_until_it_expires():
    key = ai_answer_key("cited-ticker", "AAPL", "q", day="2026-10-05")
    put_ai_answer(key, {"answer": "a", "sources": []}, now=1000.0)
    assert get_ai_answer(key, now=1000.0 + TTL_SECONDS - 1) == {"answer": "a", "sources": []}
    assert get_ai_answer(key, now=1000.0 + TTL_SECONDS + 1) is None


def test_the_cache_stays_bounded_by_dropping_the_oldest_answer():
    keys = [ai_answer_key("general", "", f"question {i}", day="2026-10-05") for i in range(MAX_ENTRIES + 1)]
    for i, key in enumerate(keys):
        put_ai_answer(key, {"answer": str(i)}, now=float(i))
    assert get_ai_answer(keys[0], now=0.0) is None
    assert get_ai_answer(keys[-1], now=float(MAX_ENTRIES)) == {"answer": str(MAX_ENTRIES)}
