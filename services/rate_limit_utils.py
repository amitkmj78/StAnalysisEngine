import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def fetch_with_backoff(
    fetch_fn: Callable[[], T],
    *,
    max_retries: int = 2,
    base_delay: float = 0.3,
    retry_delay: float = 8.0,
) -> T:
    """
    Calls fetch_fn() (a zero-arg callable wrapping one yfinance request),
    sleeping base_delay after every call to stay under Yahoo's rate limit
    in the first place, and retrying with a longer backoff specifically on
    a "Too Many Requests" response (transient, usually recovers within a
    few seconds) — a real, observed failure mode once a capture loop grows
    to hundreds of sequential per-ticker calls (e.g. the S&P 500), unlike
    the smaller ~25-ticker universes this pattern was originally written
    against. Any other exception is raised immediately, unretried — the
    caller's existing per-ticker try/except still decides whether to skip
    it, this only adds resilience to the specific rate-limit case.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            result = fetch_fn()
            time.sleep(base_delay)
            return result
        except Exception as e:
            last_exc = e
            is_rate_limited = "Too Many Requests" in str(e) or "Rate limited" in str(e)
            if is_rate_limited and attempt < max_retries:
                logger.info(
                    "Rate limited, backing off %.1fs (attempt %d/%d)",
                    retry_delay, attempt + 1, max_retries,
                )
                time.sleep(retry_delay)
                continue
            raise
    raise last_exc  # unreachable, satisfies type checkers
