"""SOC-2: the feed merges two already-real content kinds -- posts
(this batch) and community_ideas (COM-1) -- from the people/tickers/
topics a user follows, newest first. No new ranking algorithm: a plain
recency merge across heterogeneous sources, same "the service does the
pure merge, the router does the SQL" layering every other feed-shaped
module in this app uses (see web/backend/routers/alerts_inbox.py's
UNION for the same idea applied in SQL instead of Python).

Scope cut (disclosed in the tracker): the pre-existing
/strategies/shared/[token] link-sharing feature is a separate,
already-working mechanism with its own flow -- it is not retrofitted
into this feed.
"""

from __future__ import annotations


def build_feed(post_rows: list[dict], idea_rows: list[dict], limit: int = 50) -> list[dict]:
    items = [{"kind": "post", **row} for row in post_rows]
    items += [{"kind": "idea", **row} for row in idea_rows]
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items[:limit]
