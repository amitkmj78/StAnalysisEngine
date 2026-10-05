"""DIF-5: a verifiable daily record of the app's new predictions.

Each day, every score row written for that day is serialised in a fixed form and hashed. The day's
content hash covers all of them, and the chain hash also covers the previous day's chain hash, so
changing any past prediction (or removing one) changes every hash after it. Anyone holding the
published chain can recompute it and see whether the record was altered.

This module is the pure part: canonical serialisation and the chain. The database read and the
publication are kept apart from it, so the rules can be tested without a database.
"""

import hashlib
import json
from typing import Optional

FIELDS = (
    "ticker", "universe_id", "as_of_date", "short_score", "short_signal",
    "long_score", "long_signal", "sector_key", "regime",
)


def canonical_record(row: dict) -> str:
    """One prediction in a fixed form: the same values always give the same string."""
    picked = {}
    for key in FIELDS:
        value = row.get(key)
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        elif isinstance(value, float):
            value = round(value, 6)
        picked[key] = value
    return json.dumps(picked, sort_keys=True, separators=(",", ":"))


def record_hash(row: dict) -> str:
    return hashlib.sha256(canonical_record(row).encode()).hexdigest()


def day_content_hash(rows: list[dict]) -> str:
    """Hash of all of a day's predictions. Order does not matter: the records are sorted first."""
    lines = sorted(canonical_record(r) for r in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def chain_hash(previous_chain_hash: Optional[str], day: str, record_count: int, content_hash: str) -> str:
    payload = json.dumps(
        {"prev": previous_chain_hash or "", "day": day, "count": record_count, "content": content_hash},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verify_chain(entries: list[dict], rows_by_day: dict[str, list[dict]]) -> dict:
    """entries: the published chain in day order, each with day, record_count, content_hash, chain_hash.
    rows_by_day: the predictions for each day as they are stored now. Returns the first day that fails, or ok."""
    previous = None
    for entry in entries:
        rows = rows_by_day.get(entry["day"], [])
        content = day_content_hash(rows)
        if len(rows) != entry["record_count"] or content != entry["content_hash"]:
            return {"ok": False, "first_failing_day": entry["day"], "reason": "the day's predictions no longer match"}
        if chain_hash(previous, entry["day"], entry["record_count"], entry["content_hash"]) != entry["chain_hash"]:
            return {"ok": False, "first_failing_day": entry["day"], "reason": "the chain does not link"}
        previous = entry["chain_hash"]
    return {"ok": True, "days_checked": len(entries)}
