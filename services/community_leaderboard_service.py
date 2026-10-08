"""COM-4: the leaderboard's pure aggregation -- groups already-scored
community_ideas rows by author and ranks them by
services.community_idea_service.risk_adjusted_excess_return, same
"None sorts last, never claim a rank the sample doesn't support"
discipline services.challenge_leaderboard.py::build_leaderboard already
established for the challenges leaderboard. DB-free: the router does
its own query and hands this function plain row dicts, same layering
every other "service does the pure math, router does the SQL" module
in this app already uses.
"""

from __future__ import annotations

from typing import Optional

from services.community_idea_service import leaderboard_sort_key, risk_adjusted_excess_return, worst_idea

# COM-7: the app's own model appears as its own author -- same
# services.quant_model_service.MODEL_MEMBER_LABEL sentinel-identity
# convention already used for the challenges leaderboard (a synthetic
# entry, not a real users row).
from services.quant_model_service import MODEL_MEMBER_LABEL


def build_leaderboard(scored_idea_rows: list[dict], min_samples: Optional[int] = None) -> list[dict]:
    """`scored_idea_rows`: every SCORED idea (excess_vs_spy_pct is not
    None), each carrying author_user_id (None for the model author),
    is_model, display_name, direction, realized_return_pct,
    excess_vs_spy_pct. Returns one row per author, sorted best-first,
    with unscored-for-ranking (sample too small) authors last but
    still listed -- COM-4's own acceptance text ("shows return,
    volatility and sample size for each author") means an author below
    the sample gate is still visible, just not ranked by a score the
    data can't support yet.
    """
    by_author: dict[object, list[dict]] = {}
    labels: dict[object, str] = {}
    for row in scored_idea_rows:
        key = "model" if row.get("is_model") else row["author_user_id"]
        by_author.setdefault(key, []).append(row)
        labels[key] = MODEL_MEMBER_LABEL if row.get("is_model") else row.get("display_name")

    kwargs = {} if min_samples is None else {"min_samples": min_samples}
    entries = []
    for key, rows in by_author.items():
        excess_returns = [r["excess_vs_spy_pct"] for r in rows]
        score = risk_adjusted_excess_return(excess_returns, **kwargs)
        entries.append(
            {
                "author_user_id": None if key == "model" else key,
                "is_model": key == "model",
                "display_name": labels[key],
                "num_ideas": len(rows),
                "avg_excess_vs_spy_pct": round(sum(excess_returns) / len(excess_returns), 2),
                "volatility_pct": _population_stdev(excess_returns),
                "hit_rate_pct": round(sum(1 for r in rows if r["outcome"] == "hit") / len(rows) * 100, 1),
                "score": score,
                "worst_idea": worst_idea(rows),
            }
        )

    entries.sort(key=lambda e: leaderboard_sort_key(e["score"]))
    return entries


def _population_stdev(values: list[float]) -> float:
    import statistics
    return round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
