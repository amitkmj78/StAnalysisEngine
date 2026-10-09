"""SOC-8: reputation built from scored results (COM-3/COM-4), never
from follower count. Reuses services/community_leaderboard_service.py's
existing risk-adjusted score UNMODIFIED -- no second scoring path.

SOC-8's own acceptance text also counts "accepted answers (BEG-4)" --
BEG-4 is a separate, fully greenfield Q&A-forum tracker item (status:
Not started, zero code) and is not built as part of this round
(confirmed with the user). `qa_component` is hardcoded to 0 so the
total is never silently inflated by a feature that doesn't exist yet;
this is a disclosed partial reputation, not a claim of full coverage.
"""

from __future__ import annotations

from typing import Optional

from services.community_leaderboard_service import build_leaderboard

QA_COMPONENT_PLACEHOLDER = 0  # blocked on BEG-4 (accepted answers) -- not built this round


def compute_reputation(scored_idea_rows: list[dict]) -> dict:
    board = build_leaderboard(scored_idea_rows) if scored_idea_rows else []
    entry = board[0] if board else None
    track_record_component: Optional[float] = entry["score"] if entry else None
    return {
        "track_record_component": track_record_component,
        "qa_component": QA_COMPONENT_PLACEHOLDER,
        # None (not a number) until there's enough scored history to
        # support a track-record component -- same "never claim a rank
        # the data can't support" discipline as the score it wraps.
        "reputation": track_record_component,
        "num_ideas_scored": entry["num_ideas"] if entry else 0,
    }
