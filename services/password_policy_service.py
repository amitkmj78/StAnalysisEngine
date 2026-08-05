import hashlib
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MIN_LENGTH = 10
HIBP_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"


def validate_password_policy(password: str) -> Optional[str]:
    """
    Local, synchronous checks — length plus a light complexity bar (needs a
    letter AND a digit). Deliberately doesn't demand a symbol/mixed-case:
    current guidance (NIST 800-63B) finds forced character-class rules push
    people toward predictable patterns like "Password1!" without actually
    making passwords harder to guess — length is the stronger signal.
    Returns an error message, or None if the password passes.
    """
    if len(password) < MIN_LENGTH:
        return f"Password must be at least {MIN_LENGTH} characters."
    if not re.search(r"[A-Za-z]", password):
        return "Password must include at least one letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    return None


async def is_breached_password(password: str) -> bool:
    """
    Checks the password against Have I Been Pwned's Pwned Passwords list —
    the real "breach-list check" a password policy should have, not just a
    complexity regex. Uses the k-anonymity range API: only the first 5 hex
    characters of the SHA-1 hash are ever sent, so the actual password (or
    even its full hash) never leaves this server. Fails open (returns
    False, i.e. "not known-breached") if the API is unreachable — a
    third-party outage should not be able to block every signup on this
    app.
    """
    sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(HIBP_RANGE_URL.format(prefix=prefix))
            resp.raise_for_status()
    except Exception as e:
        logger.warning("Pwned Passwords check failed, failing open: %s", e)
        return False

    for line in resp.text.splitlines():
        parts = line.split(":")
        if len(parts) == 2 and parts[0].strip().upper() == suffix:
            return True
    return False
