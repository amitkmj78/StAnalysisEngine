"""Challenge notifications: daily rank, passed-you alerts, ending reminder, final result.

Every message goes through notification_dispatcher.dispatch_alert, so quiet
hours, digest mode and channel preferences apply like any other alert. Each
message is sent at most once per member, per kind, per day (challenge_notification_log).
The quant model is never sent anything, since it has no account.
"""

from datetime import date, timedelta
from typing import Optional

from services.challenge_leaderboard import build_leaderboard
from services.challenge_service import SCORING_METHODS
from services.notification_dispatcher import dispatch_alert
from web.backend.db import service_conn

ENDING_REMINDER_DAYS = 2


def rank_by_user(entries: list[dict]) -> dict[str, int]:
    """1-based rank among scored entries (model included, so the rank matches
    what the page shows). Entries with no score are not ranked."""
    scored = sorted((e for e in entries if e.get("score") is not None), key=lambda e: -e["score"])
    return {e["user_id"]: place for place, e in enumerate(scored, start=1) if e.get("user_id")}


def passed_by(prev_rank: dict[str, int], cur_rank: dict[str, int], labels: dict[str, str]) -> dict[str, str]:
    """For each member whose rank got worse, the label of the member who moved
    above them today (the strongest such mover). Pure, so it can be tested."""
    out: dict[str, str] = {}
    for user, now in cur_rank.items():
        before = prev_rank.get(user)
        if before is None or now <= before:
            continue
        movers = [(cur_rank[v], v) for v in cur_rank if v != user and cur_rank[v] < now and prev_rank.get(v, 10**6) > before]
        if movers:
            _, mover = min(movers)
            out[user] = labels.get(mover, "Another member")
    return out


def _score_text(entry: dict, scoring: str) -> str:
    if entry.get("score") is None:
        return "not scored yet"
    if scoring in ("return", "excess_spy"):
        return f"{entry['score']:+.2f}%"
    return f"{entry['score']:.2f}"


async def _claim(conn, challenge_id: int, user_id: str, kind: str, day: date) -> bool:
    """True only the first time this (challenge, member, kind, day) is claimed."""
    row = await conn.fetchval(
        """
        INSERT INTO challenge_notification_log (challenge_id, user_id, kind, as_of_date)
        VALUES ($1, $2::uuid, $3, $4) ON CONFLICT DO NOTHING RETURNING id
        """,
        challenge_id, user_id, kind, day,
    )
    return row is not None


async def _previous_ranks(conn, challenge_id: int, today: date) -> dict[str, int]:
    rows = await conn.fetch(
        """
        SELECT user_id::text, rank FROM challenge_rank_history
        WHERE challenge_id = $1 AND as_of_date = (
            SELECT max(as_of_date) FROM challenge_rank_history WHERE challenge_id = $1 AND as_of_date < $2
        )
        """,
        challenge_id, today,
    )
    return {r["user_id"]: r["rank"] for r in rows}


async def _save_ranks(conn, challenge_id: int, today: date, entries: list[dict], cur_rank: dict[str, int]) -> None:
    for e in entries:
        user = e.get("user_id")
        if not user or user not in cur_rank:
            continue
        await conn.execute(
            """
            INSERT INTO challenge_rank_history (challenge_id, user_id, as_of_date, rank, score)
            VALUES ($1, $2::uuid, $3, $4, $5) ON CONFLICT (challenge_id, user_id, as_of_date) DO NOTHING
            """,
            challenge_id, user, today, cur_rank[user], e.get("score"),
        )


async def run_challenge_notifications(today: Optional[date] = None) -> int:
    """Sends today's messages for every challenge that is running, is ending in
    ENDING_REMINDER_DAYS days, or ended yesterday. Returns the number sent."""
    today = today or date.today()
    async with service_conn() as conn:
        challenges = await conn.fetch(
            "SELECT id, name, end_date, scoring FROM challenges WHERE end_date >= $1 AND start_date <= $2",
            today - timedelta(days=1), today,
        )

    sent = 0
    for ch in challenges:
        cid, name, end_date = ch["id"], ch["name"], ch["end_date"]
        board = await build_leaderboard(cid)
        if board is None:
            continue
        entries = board["entries"]
        cur_rank = rank_by_user(entries)
        labels = {e["user_id"]: e["member"] for e in entries if e.get("user_id")}
        total = len(cur_rank)
        scoring = ch["scoring"]
        days_left = (end_date - today).days

        async with service_conn() as conn:
            prev_rank = await _previous_ranks(conn, cid, today)
            await _save_ranks(conn, cid, today, entries, cur_rank)

        passers = passed_by(prev_rank, cur_rank, labels)
        by_user = {e["user_id"]: e for e in entries if e.get("user_id")}

        for user, place in cur_rank.items():
            entry = by_user[user]
            badges = ", ".join(b["badge"] for b in entry.get("badges", []))
            if end_date == today - timedelta(days=1):
                kind = "final"
                subject = f"{name} has ended"
                body = f"You finished #{place} of {total} ({_score_text(entry, scoring)} by {SCORING_METHODS[scoring]})."
                if badges:
                    body += f" Badges: {badges}."
            elif days_left == ENDING_REMINDER_DAYS:
                kind = "ending"
                subject = f"{name} ends in {days_left} days"
                body = f"You're #{place} of {total} with {_score_text(entry, scoring)}. Check the leaderboard before it closes."
            else:
                kind = "passed" if user in passers else "daily"
                if kind == "passed":
                    subject = f"{passers[user]} passed you in {name}"
                    body = f"You're now #{place} of {total} in {name}."
                else:
                    subject = f"{name}: you're #{place} of {total} today"
                    body = f"Score {_score_text(entry, scoring)} by {SCORING_METHODS[scoring]}."

            async with service_conn() as conn:
                claimed = await _claim(conn, cid, user, kind, today)
            if claimed:
                await dispatch_alert(user, None, "challenge_update", subject, body)
                sent += 1

    return sent
