"""SAF-3: automated flags for sudden spikes in mentions of small or
illiquid stocks, and for repeated identical posts across different
accounts. Flags are held for admin review (flagged_content), never
auto-removed -- same "flag, don't silently act" posture
community_ideas' auto-hide-at-threshold already uses for SAF-2.

Gated behind app_settings.COMMUNITY_SAFETY_SCAN_ENABLED_KEY, default off
-- a moderation job must not start flagging real users' content the
moment this deploys.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import yfinance as yf
from starlette.concurrency import run_in_threadpool

from web.backend.db import service_conn

logger = logging.getLogger(__name__)

MENTION_SPIKE_LOOKBACK_DAYS = 7
# A ticker needs a real minimum volume of mentions today before it's even
# considered -- otherwise one extra mention on a ticker nobody talks about
# would count as an "infinite" spike over a near-zero baseline.
MENTION_SPIKE_MIN_TODAY = 5
MENTION_SPIKE_RATIO = 3.0  # today's mentions must be at least this many times the trailing daily average
SMALL_CAP_CEILING_USD = 2_000_000_000  # "small or illiquid", per SAF-3's own wording
DUPLICATE_WINDOW_HOURS = 24
MIN_DUPLICATE_BODY_LENGTH = 20  # skip trivially short posts ("lol", "nice") -- too noisy to mean coordination


async def _get_market_cap(ticker: str) -> float | None:
    def _fetch() -> float | None:
        try:
            cap = yf.Ticker(ticker).fast_info.get("market_cap")
            return float(cap) if cap else None
        except Exception:  # noqa: BLE001 -- one bad ticker lookup must never abort the scan
            return None
    return await run_in_threadpool(_fetch)


async def detect_mention_spikes() -> int:
    """Flags (ticker, today) pairs where today's post-mention count is both
    above an absolute floor and a real multiple of the trailing 7-day daily
    average, restricted to small/illiquid tickers. A rare double-flag on a
    scheduler misfire retry is accepted -- the admin queue just shows the
    same ticker twice, never a wrongful action taken on anyone's content."""
    today = date.today()
    lookback_start = today - timedelta(days=MENTION_SPIKE_LOOKBACK_DAYS)
    async with service_conn() as conn:
        today_rows = await conn.fetch(
            "SELECT ticker, count(*) AS n FROM posts WHERE ticker IS NOT NULL AND created_at::date = $1 GROUP BY ticker",
            today,
        )
        if not today_rows:
            return 0
        trailing_rows = await conn.fetch(
            """
            SELECT ticker, count(*) AS n FROM posts
            WHERE ticker IS NOT NULL AND created_at::date >= $1 AND created_at::date < $2
            GROUP BY ticker
            """,
            lookback_start, today,
        )
    trailing_by_ticker = {r["ticker"]: r["n"] for r in trailing_rows}

    flagged = 0
    for row in today_rows:
        ticker, today_count = row["ticker"], row["n"]
        if today_count < MENTION_SPIKE_MIN_TODAY:
            continue
        trailing_avg = trailing_by_ticker.get(ticker, 0) / MENTION_SPIKE_LOOKBACK_DAYS
        if trailing_avg > 0 and today_count < trailing_avg * MENTION_SPIKE_RATIO:
            continue
        market_cap = await _get_market_cap(ticker)
        if market_cap is None or market_cap > SMALL_CAP_CEILING_USD:
            continue
        detail = (
            f"{ticker}: {today_count} mentions today vs a {trailing_avg:.1f}/day trailing average "
            f"(market cap ~${market_cap / 1_000_000_000:.2f}B)"
        )
        async with service_conn() as conn:
            await conn.execute(
                "INSERT INTO flagged_content (content_type, flag_type, detail) VALUES ('ticker_mentions', 'mention_spike', $1)",
                detail,
            )
        flagged += 1
    return flagged


async def detect_duplicate_posts() -> int:
    """Near-identical post bodies (normalized whitespace/case) from
    DIFFERENT authors within a rolling 24h window -- the same author
    posting the same thing twice isn't coordination."""
    window_start = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_WINDOW_HOURS)
    async with service_conn() as conn:
        rows = await conn.fetch(
            "SELECT id, author_user_id, body FROM posts WHERE created_at >= $1 AND NOT hidden",
            window_start,
        )
        already_flagged = {
            r["content_id"] for r in await conn.fetch(
                "SELECT content_id FROM flagged_content WHERE flag_type = 'duplicate_post' AND created_at >= $1",
                window_start,
            )
        }

    groups: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        key = " ".join(r["body"].strip().lower().split())
        if len(key) < MIN_DUPLICATE_BODY_LENGTH:
            continue
        groups.setdefault(key, []).append((r["id"], str(r["author_user_id"])))

    flagged = 0
    for posts in groups.values():
        distinct_authors = {author for _, author in posts}
        if len(distinct_authors) < 2:
            continue
        for post_id, _author in posts:
            if post_id in already_flagged:
                continue
            async with service_conn() as conn:
                await conn.execute(
                    """
                    INSERT INTO flagged_content (content_type, content_id, flag_type, detail)
                    VALUES ('post', $1, 'duplicate_post', $2)
                    """,
                    post_id,
                    f"Identical body posted by {len(distinct_authors)} different accounts within {DUPLICATE_WINDOW_HOURS}h",
                )
            flagged += 1
    return flagged


async def run_community_safety_scan() -> dict:
    spikes = 0
    duplicates = 0
    try:
        spikes = await detect_mention_spikes()
    except Exception as e:  # noqa: BLE001 -- one detector failing must not block the other
        logger.warning("Mention-spike detection failed: %s", e)
    try:
        duplicates = await detect_duplicate_posts()
    except Exception as e:  # noqa: BLE001
        logger.warning("Duplicate-post detection failed: %s", e)
    return {"mention_spikes": spikes, "duplicate_posts": duplicates}
