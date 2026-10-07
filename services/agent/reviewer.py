"""AI reviewer (AGT-21..24): reads the plan's new-entry buy candidates
plus each one's recent, dated headlines, and may remove -- never add or
resize -- a buy, with a reason grounded in one of those real headlines.

Deliberately scoped so it structurally cannot see sells, stops, or
rebalances (AGT-22): the caller (services/agent/runner.py) only ever
passes this module the new-entry buy subset of a run's proposed orders.
Everything here is pure except the two I/O calls (headline fetch, LLM
call); all journaling is the caller's responsibility, same separation
services/agent/risk.py's pure functions already use.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from starlette.concurrency import run_in_threadpool

from services.agent.risk import Order
from services.llm_setup import invoke_with_fallback
from services.yfinance_cache import get_cached_ticker_news

logger = logging.getLogger(__name__)

MAX_HEADLINES_PER_TICKER = 8


@dataclass
class ReviewOutcome:
    kept: list[Order]
    removed: list[dict] = field(default_factory=list)  # [{ticker, date, headline, reason}]
    ignored: list[dict] = field(default_factory=list)  # [{..., why_ignored}]
    skipped: bool = False
    skip_reason: Optional[str] = None


def _format_headlines(ticker: str, headlines: list[dict]) -> str:
    if not headlines:
        return f"{ticker}: no recent headlines found."
    lines = [f'- {h["published_at"] or "undated"}: "{h["title"]}"' for h in headlines[:MAX_HEADLINES_PER_TICKER]]
    return f"{ticker} recent headlines:\n" + "\n".join(lines)


def _build_prompt(candidates: list[Order], headlines_by_ticker: dict[str, list[dict]]) -> str:
    sections = "\n\n".join(_format_headlines(o.ticker, headlines_by_ticker.get(o.ticker, [])) for o in candidates)
    tickers = ", ".join(o.ticker for o in candidates)
    return (
        "You are reviewing a list of proposed new stock purchases for a rule-based trading system. "
        f"The tickers under review are exactly: {tickers}. You may ONLY recommend REMOVING a ticker "
        "from this list; you may never add a ticker, increase its size, or act on anything else. Base "
        "every removal strictly on one of the real headlines given below for that exact ticker -- quote "
        "the headline text and its date exactly as given, character for character. If nothing below "
        "justifies removing a ticker, say nothing about it.\n\n"
        f"{sections}\n\n"
        "Respond with one line per removal, in exactly this format, and nothing else:\n"
        "REMOVE <TICKER> | <date exactly as given above> | <headline text exactly as given above> | <one-sentence reason>\n"
        "If no ticker should be removed, respond with exactly: NONE"
    )


def _parse_response(text: str, candidate_tickers: set[str]) -> tuple[list[dict], list[dict]]:
    """Returns (parsed_removal_requests, rejected_lines) -- rejected
    covers anything malformed or naming a ticker outside today's
    candidate set (AGT-21's "any attempt to add or enlarge is ignored")."""
    requests: list[dict] = []
    rejected: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper() == "NONE":
            continue
        if not line.upper().startswith("REMOVE "):
            rejected.append({"raw_line": line, "why_ignored": "Not a REMOVE instruction; only removals are allowed."})
            continue
        parts = [p.strip() for p in line[len("REMOVE "):].split("|")]
        if len(parts) != 4:
            rejected.append({"raw_line": line, "why_ignored": "Malformed REMOVE line (expected 4 fields)."})
            continue
        ticker, date, headline, reason = parts
        ticker = ticker.upper()
        if ticker not in candidate_tickers:
            rejected.append({"raw_line": line, "ticker": ticker,
                             "why_ignored": f"{ticker} is not one of today's proposed buys; ignored."})
            continue
        requests.append({"ticker": ticker, "date": date, "headline": headline, "reason": reason})
    return requests, rejected


def _grounded(request: dict, headlines_by_ticker: dict[str, list[dict]]) -> bool:
    """AGT-24: the cited headline and date must actually be among what
    was fed to the model for that ticker -- a model can be instructed to
    quote its source, but only code can verify it didn't just invent one."""
    cited_headline = request["headline"].strip().lower()
    cited_date = request["date"].strip()
    for h in headlines_by_ticker.get(request["ticker"], []):
        if h["title"].strip().lower() == cited_headline and (h["published_at"] or "undated") == cited_date:
            return True
    return False


async def review_new_entries(candidates: list[Order], llms: list, timeout_seconds: float) -> ReviewOutcome:
    if not candidates:
        return ReviewOutcome(kept=[])

    headlines_by_ticker = {
        o.ticker: await run_in_threadpool(get_cached_ticker_news, o.ticker) for o in candidates
    }

    try:
        prompt = _build_prompt(candidates, headlines_by_ticker)
        content, _ = await asyncio.wait_for(
            run_in_threadpool(invoke_with_fallback, llms, prompt), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return ReviewOutcome(kept=candidates, skipped=True, skip_reason=f"AI review timed out after {timeout_seconds:g}s.")
    except Exception as e:  # noqa: BLE001 -- AGT-23: any failure must fail open, never block the deterministic plan
        logger.warning("AI reviewer failed; proceeding with the unreviewed plan: %s", e)
        return ReviewOutcome(kept=candidates, skipped=True, skip_reason=f"AI review failed: {e}")

    candidate_tickers = {o.ticker for o in candidates}
    requests, ignored = _parse_response(content, candidate_tickers)

    removed_tickers: set[str] = set()
    removed: list[dict] = []
    for req in requests:
        if _grounded(req, headlines_by_ticker):
            removed_tickers.add(req["ticker"])
            removed.append(req)
        else:
            ignored.append({**req, "why_ignored": "Cited headline/date not found in the real headlines fetched for this ticker; removal rejected."})

    kept = [o for o in candidates if o.ticker not in removed_tickers]
    return ReviewOutcome(kept=kept, removed=removed, ignored=ignored, skipped=False)
