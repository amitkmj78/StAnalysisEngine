"""SOC-8: reputation built from scored results (COM-3/COM-4) plus
accepted Q&A answers (BEG-4), never from follower count. Reuses
services/community_leaderboard_service.py's existing risk-adjusted score
UNMODIFIED -- no second scoring path.

BEG-4 was a separate, greenfield Q&A-forum tracker item deferred in an
earlier round; `qa_component` was a hardcoded 0 placeholder until BEG-4
shipped. Now it's a real, disclosed count of accepted answers -- not a
weighted score, since there's no principled weight to assign an accepted
answer against a scored-idea return without fabricating one.
"""

from __future__ import annotations

from typing import Optional

from services.community_leaderboard_service import build_leaderboard


def compute_reputation(scored_idea_rows: list[dict], accepted_answer_count: int = 0) -> dict:
    board = build_leaderboard(scored_idea_rows) if scored_idea_rows else []
    entry = board[0] if board else None
    track_record_component: Optional[float] = entry["score"] if entry else None
    return {
        "track_record_component": track_record_component,
        "qa_component": max(0, accepted_answer_count),
        # None (not a number) until there's enough scored history to
        # support a track-record component -- same "never claim a rank
        # the data can't support" discipline as the score it wraps.
        # qa_component is reported separately, not folded in here, since
        # the two aren't on a comparable scale.
        "reputation": track_record_component,
        "num_ideas_scored": entry["num_ideas"] if entry else 0,
    }
