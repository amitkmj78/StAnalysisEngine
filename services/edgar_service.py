"""
SUM-1: real SEC filing data, straight from SEC EDGAR's free developer APIs --
no paid dependency, nothing LLM-guessed. See services/filing_summary_service.py
for the diffing/summarization layer built on top of this.

Section extraction here is a disclosed approximation, not a precise one: a
real 10-K/10-Q's HTML has no filer-provided anchor that identifies "this is
Item 7" -- verified directly against a real Apple 10-K while designing this
(no <span id>, and every <div id> is an opaque iXBRL tag id unrelated to
section headings). Each "Item N." heading reliably appears twice in the
extracted text (once in the Table of Contents, once at the real section) --
extract_item_sections below takes the LAST occurrence of each heading as the
real section start. Good enough to pull real section text for summarization;
not good enough to deep-link into a specific paragraph, which is why "section
links" in the UI mean "link to the real filing document, with the Item named
in the summary text" rather than a working in-document anchor.
"""

import logging
import os
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from .cache_utils import ttl_cache

logger = logging.getLogger(__name__)

# SEC's access policy requires every automated request to self-identify with
# a real app/company name and contact email -- not optional, and they may
# block unidentified bots. Read from env rather than hardcoded so it's easy
# to change; if unset, callers must treat EDGAR as unavailable rather than
# send an unidentified request.
EDGAR_CONTACT_EMAIL = os.getenv("EDGAR_CONTACT_EMAIL")
EDGAR_USER_AGENT = f"StAnalysisEngine {EDGAR_CONTACT_EMAIL}" if EDGAR_CONTACT_EMAIL else None

REQUEST_TIMEOUT_SECONDS = 15.0
TICKER_MAP_TTL_SECONDS = 60 * 60 * 24  # company_tickers.json only changes with new listings
FILINGS_LIST_TTL_SECONDS = 60 * 60 * 6  # reflects *today's* new filings, refreshed a few times a day

# Each extracted section is capped so the summarization prompt stays a
# reasonable size -- real MD&A/Risk Factors sections can run tens of
# thousands of characters. A diff prompt includes up to 4 of these blocks
# (2 items x current+previous) -- live-tested against this app's actual
# configured LLM providers and found that even the "cheap" free-tier Groq
# model this app falls back to caps requests at 8000 tokens/minute; 15000
# chars/section produced a real ~13,000-token prompt that was rejected
# every time. 4000 chars (matching services/web_search/extract.py's own
# MAX_CONTENT_CHARS) keeps 4 blocks comfortably under that limit.
# Truncation is reported by the caller, not hidden.
MAX_SECTION_CHARS = 4000

# item label -> a short marker from that item's SEC-standardized canonical
# title (Regulation S-K uses the same item titles across virtually every
# filer). The marker matters, not just the item number: verified against a
# real Apple 10-K during design that "Item 7" alone appears 4 times --
# the Table of Contents, the real heading, and two prose cross-references
# elsewhere in the document ("Item 7 of this Form 10-K under the heading...").
# Anchoring on the title immediately after the number reliably narrows that
# down to exactly 2 matches (TOC + real heading), which is what the
# last-occurrence heuristic in extract_item_sections depends on. Item 2
# means something different on a 10-K (Properties) than a 10-Q (MD&A), so
# this is scoped per form type, not a flat item-number map.
FORM_SECTIONS = {
    "10-K": {"Item 1A": "Risk Factors", "Item 7": "Management"},
    "10-Q": {"Item 1A": "Risk Factors", "Item 2": "Management"},
}


def _headers() -> dict:
    if not EDGAR_USER_AGENT:
        raise RuntimeError(
            "EDGAR_CONTACT_EMAIL is not configured -- refusing to send an unidentified "
            "request to SEC EDGAR (their access policy requires a real identifying User-Agent)."
        )
    return {"User-Agent": EDGAR_USER_AGENT}


@ttl_cache(maxsize=1, ttl_seconds=TICKER_MAP_TTL_SECONDS)
def _fetch_ticker_to_cik_map() -> dict[str, str]:
    """{"AAPL": "0000320193", ...} -- zero-padded to 10 digits, SEC's own
    convention for CIK strings used in their other endpoints' URLs."""
    response = httpx.get(
        "https://www.sec.gov/files/company_tickers.json", headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()
    return {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()}


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Many tickers (funds/ETFs, mostly) aren't SEC-registered operating
    companies and won't resolve -- returns None, same honest-omission
    precedent as services.portfolio_review_service.compute_sectors."""
    try:
        return _fetch_ticker_to_cik_map().get(ticker.upper())
    except Exception as e:
        logger.warning("edgar: failed to fetch/parse company_tickers.json: %s", e)
        return None


@ttl_cache(maxsize=256, ttl_seconds=FILINGS_LIST_TTL_SECONDS)
def get_recent_filings(cik: str, form_types: tuple = ("10-K", "10-Q")) -> list[dict]:
    """Real filing history for a company, straight from SEC's own submissions
    JSON -- [{"form", "accession_number", "filing_date", "report_date",
    "primary_document"}, ...], filtered to form_types, newest first."""
    response = httpx.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]

    filings = []
    for i, form in enumerate(recent["form"]):
        if form not in form_types:
            continue
        filings.append(
            {
                "form": form,
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "primary_document": recent["primaryDocument"][i],
            }
        )
    filings.sort(key=lambda f: f["filing_date"], reverse=True)
    return filings


def filing_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_no_dashes = accession_number.replace("-", "")
    cik_no_leading_zeros = str(int(cik))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_leading_zeros}/{accession_no_dashes}/{primary_document}"


def fetch_filing_document_text(cik: str, accession_number: str, primary_document: str) -> Optional[str]:
    """Raw HTML of the actual filing document. Returns None on any failure
    (network, 404, etc.) -- caller skips this filing, retried on the next
    scheduled run rather than ever showing a broken partial summary."""
    url = filing_document_url(cik, accession_number, primary_document)
    try:
        response = httpx.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.warning("edgar: failed to fetch filing document %s: %s", url, e)
        return None


def _heading_pattern(item: str, marker: str) -> re.Pattern:
    # "Item 1A", "Risk Factors" -> matches "Item 1A.<whitespace>Risk Factors",
    # case-insensitive, tolerant of the nbsp runs (and, in the Table of
    # Contents, the newline) real EDGAR HTML uses between the item number
    # and its title.
    return re.compile(re.escape(item) + r"\.?[\s\xa0]*" + re.escape(marker), re.IGNORECASE)


def extract_item_sections(html: str, items: dict[str, str]) -> dict[str, str]:
    """Real section text per Item. `items`: {"Item 7": "Management", ...} --
    label -> the canonical-title marker to anchor on (see FORM_SECTIONS).
    An Item not found in the document (filers vary, and not every filing
    has every item) is simply omitted -- never a fabricated "no content"
    placeholder.

    Heuristic (see module docstring): each real item heading, anchored on
    its canonical title, appears exactly twice (Table of Contents, then the
    real section) -- the LAST match in document order is treated as the
    real section start. Section content runs to the next found item's
    last-match position, or to the end of the document for the final item.
    Truncated to MAX_SECTION_CHARS; the caller (services.filing_summary_
    service) is responsible for disclosing when that cap was hit, since
    this function only returns the (possibly truncated) text, not a
    truncation flag -- callers compare len() against MAX_SECTION_CHARS
    themselves.
    """
    text = BeautifulSoup(html, "html.parser").get_text("\n")

    positions: dict[str, int] = {}
    for item, marker in items.items():
        matches = list(_heading_pattern(item, marker).finditer(text))
        if matches:
            positions[item] = matches[-1].start()

    ordered = sorted(positions.items(), key=lambda kv: kv[1])
    sections: dict[str, str] = {}
    for idx, (item, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(text)
        sections[item] = text[start:end].strip()[:MAX_SECTION_CHARS]
    return sections
