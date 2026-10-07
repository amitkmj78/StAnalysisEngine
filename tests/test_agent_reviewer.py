import asyncio

import pytest

from services.agent import reviewer
from services.agent.risk import Order


def _buy(ticker: str) -> Order:
    return Order(ticker, "buy", 10, 100.0, 1000.0, "new_entry", "New Buy signal passed all filters.")


def _headlines(ticker: str):
    return {ticker: [{"title": "Company announces pending lawsuit over patent dispute", "published_at": "2026-10-05"}]}


def _run(coro):
    return asyncio.run(coro)


# --- AGT-21: removal-only, out-of-scope instructions ignored and logged ---

def test_grounded_removal_is_applied(monkeypatch):
    candidates = [_buy("AAA"), _buy("BBB")]
    headlines = {**_headlines("AAA"), "BBB": []}
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: headlines.get(t, []))

    response = 'REMOVE AAA | 2026-10-05 | Company announces pending lawsuit over patent dispute | Pending litigation risk.'
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: (response, 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert [o.ticker for o in outcome.kept] == ["BBB"]
    assert len(outcome.removed) == 1
    assert outcome.removed[0]["ticker"] == "AAA"
    assert not outcome.skipped


def test_attempt_to_act_on_a_non_candidate_ticker_is_ignored_not_applied(monkeypatch):
    candidates = [_buy("AAA")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: _headlines("AAA").get(t, []))
    # "ZZZ" was never one of today's proposed buys.
    response = "REMOVE ZZZ | 2026-10-05 | Some headline | Some reason."
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: (response, 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert [o.ticker for o in outcome.kept] == ["AAA"]  # untouched
    assert outcome.removed == []
    assert len(outcome.ignored) == 1
    assert "ZZZ" in outcome.ignored[0]["why_ignored"]


def test_malformed_line_is_ignored_not_applied(monkeypatch):
    candidates = [_buy("AAA")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: [])
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: ("REMOVE AAA because I feel like it", 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert [o.ticker for o in outcome.kept] == ["AAA"]
    assert outcome.removed == []
    assert len(outcome.ignored) == 1


def test_none_response_removes_nothing(monkeypatch):
    candidates = [_buy("AAA"), _buy("BBB")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: [])
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: ("NONE", 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert {o.ticker for o in outcome.kept} == {"AAA", "BBB"}
    assert outcome.removed == outcome.ignored == []


# --- AGT-24: grounding -- a cited headline/date must be real ---

def test_fabricated_headline_citation_is_rejected_not_applied(monkeypatch):
    candidates = [_buy("AAA")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: _headlines("AAA").get(t, []))
    # Headline text doesn't match anything actually fetched for AAA.
    response = "REMOVE AAA | 2026-10-05 | CEO resigns amid fraud investigation | Governance risk."
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: (response, 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert [o.ticker for o in outcome.kept] == ["AAA"]
    assert outcome.removed == []
    assert len(outcome.ignored) == 1
    assert "not found" in outcome.ignored[0]["why_ignored"]


def test_wrong_date_on_a_real_headline_is_rejected_not_applied(monkeypatch):
    candidates = [_buy("AAA")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: _headlines("AAA").get(t, []))
    # Real headline text, but a date that doesn't match what was actually given (2026-10-05).
    response = 'REMOVE AAA | 2026-09-01 | Company announces pending lawsuit over patent dispute | Litigation risk.'
    monkeypatch.setattr(reviewer, "invoke_with_fallback", lambda llms, prompt: (response, 0))

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert outcome.removed == []
    assert len(outcome.ignored) == 1


# --- AGT-23: fail open on timeout or any LLM failure ---

def test_llm_timeout_fails_open_with_plan_unchanged(monkeypatch):
    candidates = [_buy("AAA"), _buy("BBB")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: [])

    def _hangs(llms, prompt):
        import time
        time.sleep(1.0)
        return "NONE", 0

    monkeypatch.setattr(reviewer, "invoke_with_fallback", _hangs)

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=0.05))
    assert outcome.skipped is True
    assert "timed out" in outcome.skip_reason
    assert {o.ticker for o in outcome.kept} == {"AAA", "BBB"}
    assert outcome.removed == []


def test_llm_exception_fails_open_with_plan_unchanged(monkeypatch):
    candidates = [_buy("AAA")]
    monkeypatch.setattr(reviewer, "get_cached_ticker_news", lambda t: [])

    def _boom(llms, prompt):
        raise RuntimeError("every provider failed")

    monkeypatch.setattr(reviewer, "invoke_with_fallback", _boom)

    outcome = _run(reviewer.review_new_entries(candidates, ["fake-llm"], timeout_seconds=5.0))
    assert outcome.skipped is True
    assert "failed" in outcome.skip_reason
    assert [o.ticker for o in outcome.kept] == ["AAA"]


def test_no_candidates_is_a_no_op():
    outcome = _run(reviewer.review_new_entries([], ["fake-llm"], timeout_seconds=5.0))
    assert outcome.kept == []
    assert outcome.skipped is False
