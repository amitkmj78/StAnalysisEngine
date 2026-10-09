from datetime import datetime, timedelta

from services.social_feed_service import build_feed


def test_build_feed_merges_posts_and_ideas_by_recency():
    now = datetime(2026, 10, 1, 12, 0)
    posts = [{"id": 1, "body": "note", "created_at": now - timedelta(hours=2)}]
    ideas = [{"id": 1, "ticker": "AAPL", "created_at": now}]

    feed = build_feed(posts, ideas)

    assert [item["kind"] for item in feed] == ["idea", "post"]
    assert feed[0]["created_at"] == now


def test_build_feed_respects_limit():
    now = datetime(2026, 10, 1, 12, 0)
    posts = [{"id": i, "created_at": now - timedelta(minutes=i)} for i in range(5)]

    feed = build_feed(posts, [], limit=2)

    assert len(feed) == 2
    assert feed[0]["id"] == 0


def test_build_feed_empty_sources_returns_empty():
    assert build_feed([], []) == []
