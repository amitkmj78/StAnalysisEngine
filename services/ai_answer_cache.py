"""NFR-6: cache AI answers per stock per day, so repeat questions don't call the model again.

The key is built from the kind of answer, the stock (or "general"), the UTC date and the question with
case and spacing normalised. A new day gives a new key, so an answer is never older than one day. The
cache lives in this process only and holds a bounded number of answers.
"""

import hashlib
import threading
import time
from datetime import datetime, timezone
from typing import Optional

MAX_ENTRIES = 500
TTL_SECONDS = 24 * 60 * 60

_store: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def ai_answer_key(kind: str, subject: str, question: str, day: Optional[str] = None) -> str:
    day = day or datetime.now(timezone.utc).date().isoformat()
    normalized = " ".join(question.lower().split())
    raw = f"{kind}|{subject.strip().upper()}|{day}|{normalized}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_ai_answer(key: str, now: Optional[float] = None) -> Optional[dict]:
    now = time.time() if now is None else now
    with _lock:
        item = _store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= now:
            _store.pop(key, None)
            return None
        return dict(value)


def put_ai_answer(key: str, value: dict, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    with _lock:
        if key not in _store and len(_store) >= MAX_ENTRIES:
            oldest = min(_store, key=lambda k: _store[k][0])
            _store.pop(oldest, None)
        _store[key] = (now + TTL_SECONDS, dict(value))


def clear_ai_answers() -> None:
    with _lock:
        _store.clear()
