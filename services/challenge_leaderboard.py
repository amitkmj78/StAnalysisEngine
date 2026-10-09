"""Leaderboard for a challenge, shared by the page and the notification job.

Entries carry an internal user_id for the notification job. The API strips it
before responding, so a member's identity is only ever the masked label.
"""

from datetime import date
from typing import Optional

from starlette.concurrency import run_in_threadpool

from services.challenge_service import (
    DIVERSIFICATION_CONCENTRATION_LIMIT_PCT,
    MIN_DAYS_FOR_RISK_SCORE,
    SCORING_METHODS,
    compute_member_diversification,
    compute_member_performance,
    mask_email,
    score_for,
)
from services.quant_model_service import MODEL_MEMBER_LABEL, model_snapshots
from services.yfinance_cache import get_cached_history
from web.backend.db import service_conn


async def _spy_return_pct(start_date: date, end_date: date) -> Optional[float]:
    """SPY over the same window the members are measured on: first close on or
    after start, last close on or before end. None if price data is missing,
    so excess-return columns go blank instead of guessing."""
    try:
        closes = (await run_in_threadpool(get_cached_history, "SPY", "2y", True))["Close"]
    except Exception:  # noqa: BLE001 -- benchmark is context; never fail the leaderboard over it
        return None
    window = closes.loc[str(start_date): str(end_date)]
    if len(window) < 2:
        return None
    return round((float(window.iloc[-1]) / float(window.iloc[0]) - 1.0) * 100.0, 2)


def compute_badges(entries: list[dict], ended: bool) -> dict[int, list[dict]]:
    """Badges keyed by entry index. Only real members can earn them, not the
    model. Lowest drawdown and most consistent need the full risk-score history,
    so they are never awarded on thin data. While a challenge runs, badges mean
    'currently leading' and are labelled that way."""
    status = "final" if ended else "current"
    out: dict[int, list[dict]] = {i: [] for i in range(len(entries))}
    humans = [(i, e) for i, e in enumerate(entries) if not e.get("is_model")]

    ranked = [(i, e) for i, e in humans if e.get("score") is not None]
    for place, (i, e) in enumerate(ranked[:3], start=1):
        out[i].append({"badge": f"Top {place}", "detail": f"#{place} by score ({status})"})

    risk_ok = [(i, e) for i, e in humans if e.get("days_of_data", 0) >= MIN_DAYS_FOR_RISK_SCORE]

    dd = [(i, e) for i, e in risk_ok if e.get("max_drawdown_pct") is not None]
    if dd:
        best = min(abs(e["max_drawdown_pct"]) for _, e in dd)
        for i, e in dd:
            if abs(e["max_drawdown_pct"]) == best:
                out[i].append({"badge": "Lowest Drawdown", "detail": f"{e['max_drawdown_pct']}% max drawdown ({status})"})

    vol = [(i, e) for i, e in risk_ok if e.get("annualized_volatility_pct") is not None]
    if vol:
        calmest = min(e["annualized_volatility_pct"] for _, e in vol)
        for i, e in vol:
            if e["annualized_volatility_pct"] == calmest:
                out[i].append({"badge": "Most Consistent", "detail": f"{calmest}% annualized volatility ({status})"})
    return out


async def build_leaderboard(challenge_id: int) -> Optional[dict]:
    async with service_conn() as conn:
        challenge = await conn.fetchrow(
            "SELECT start_date, end_date, scoring, include_quant_model FROM challenges WHERE id = $1", challenge_id
        )
        if challenge is None:
            return None
        member_rows = await conn.fetch(
            """
            SELECT m.user_id, u.email, a.id AS alpaca_paper_account_id,
                   a.api_key_id, a.api_secret_key_encrypted
            FROM challenge_members m
            JOIN users u ON u.id = m.user_id
            LEFT JOIN alpaca_paper_accounts a ON a.user_id = m.user_id
            WHERE m.challenge_id = $1
            """,
            challenge_id,
        )
        start_date = challenge["start_date"]
        end_date = min(challenge["end_date"], date.today())

        entries: list[dict] = []
        # Parallel to `entries` -- the account dict for Alpaca calls made
        # AFTER this connection closes (diversification only, below),
        # same "don't hold a DB connection open across a slow external
        # call" shape as everywhere else in this module.
        entry_accounts: list[Optional[dict]] = []
        for member in member_rows:
            base = {"user_id": str(member["user_id"]), "member": mask_email(member["email"]), "is_model": False}
            if member["alpaca_paper_account_id"] is None:
                entries.append({**base, "has_paper_account": False, "return_pct": None, "max_drawdown_pct": None,
                                "annualized_volatility_pct": None, "days_of_data": 0})
                entry_accounts.append(None)
                continue
            snapshot_rows = await conn.fetch(
                """
                SELECT as_of_date, equity FROM paper_account_equity_snapshots
                WHERE alpaca_paper_account_id = $1 AND as_of_date BETWEEN $2 AND $3
                """,
                member["alpaca_paper_account_id"], start_date, end_date,
            )
            performance = compute_member_performance(
                [{"as_of_date": r["as_of_date"], "equity": r["equity"]} for r in snapshot_rows],
                start_date, end_date,
            )
            entries.append({**base, "has_paper_account": True, **performance})
            entry_accounts.append({
                "id": member["alpaca_paper_account_id"],
                "api_key_id": member["api_key_id"],
                "api_secret_key_encrypted": member["api_secret_key_encrypted"],
            })

    if challenge["include_quant_model"]:
        model_perf = compute_member_performance(await model_snapshots(start_date, end_date), start_date, end_date)
        entries.append({"user_id": None, "member": MODEL_MEMBER_LABEL, "is_model": True,
                        "has_paper_account": True, **model_perf})
        entry_accounts.append(None)  # the model has no real Alpaca account to check

    spy_return = await _spy_return_pct(start_date, end_date)
    method = challenge["scoring"]
    for e in entries:
        e["vs_spy_pct"] = (
            round(e["return_pct"] - spy_return, 2)
            if e.get("return_pct") is not None and spy_return is not None else None
        )
        e["score"] = score_for(method, e, spy_return)

    # BEG-6: diversification only costs an Alpaca call for challenges that
    # actually use it -- every other scoring method is unaffected.
    if method == "diversified":
        for e, account in zip(entries, entry_accounts):
            if account is None:
                e["largest_position_pct"] = None
                e["diversification_ok"] = None
                e["holdings_count"] = None
                continue
            e.update(await compute_member_diversification(account))

        entries.sort(
            key=lambda e: (
                e.get("diversification_ok") is not True,
                e["score"] is None,
                -(e["score"] or 0),
            )
        )
    else:
        entries.sort(key=lambda e: (e["score"] is None, -(e["score"] or 0)))
    badges = compute_badges(entries, ended=challenge["end_date"] < date.today())
    for i, e in enumerate(entries):
        e["badges"] = badges[i]

    return {
        "start_date": str(start_date), "end_date": str(end_date),
        "scoring": method, "scoring_label": SCORING_METHODS[method],
        "spy_return_pct": spy_return, "ended": challenge["end_date"] < date.today(),
        "diversification_limit_pct": DIVERSIFICATION_CONCENTRATION_LIMIT_PCT if method == "diversified" else None,
        "entries": entries,
    }
