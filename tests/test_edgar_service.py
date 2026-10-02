from unittest.mock import patch

import pytest

from services.edgar_service import (
    MAX_SECTION_CHARS,
    extract_item_sections,
    filing_document_url,
    find_exhibit_991_document,
    get_cik_for_ticker,
    get_recent_8k_filings,
)

# A trimmed reproduction of the real pattern verified against a live Apple
# 10-K while designing this: each "Item N." heading appears once in the
# Table of Contents (clustered, no real content between them), once at the
# real section (followed by substantial real content), AND -- the bug this
# fixture guards against -- sometimes again as a bare prose cross-reference
# elsewhere in the document ("as described in Item 7 of this Form 10-K"),
# which a naive "last occurrence of the item number" heuristic would
# wrongly treat as the real section start. Anchoring on each item's
# canonical title (see FORM_SECTIONS) is what tells the real heading apart
# from a cross-reference, since a cross-reference is never immediately
# followed by the canonical title text.
_ITEMS = {"Item 1A": "Risk Factors", "Item 7": "Management"}

_SAMPLE_10K_TEXT = """
TABLE OF CONTENTS
Item 1A.
Risk Factors
Item 7.
Management's Discussion and Analysis
Item 7A.
Quantitative and Qualitative Disclosures About Market Risk

PART I
Item 1A.
Risk Factors
The Company is subject to macroeconomic conditions that could materially harm the business.
Global supply chain disruptions could increase costs. See Item 7 of this Form 10-K for further detail.
PART II
Item 7.
Management's Discussion and Analysis
Revenue increased 8% year over year, driven by strong iPhone demand.
Gross margin improved 150 basis points on favorable mix.
Item 7A.
Quantitative and Qualitative Disclosures About Market Risk
The Company is exposed to foreign currency and interest rate risk. See Item 7 of the Annual Report.
"""


def _html(text: str) -> str:
    return f"<html><body><pre>{text}</pre></body></html>"


def test_extract_item_sections_uses_last_occurrence_not_toc():
    sections = extract_item_sections(_html(_SAMPLE_10K_TEXT), _ITEMS)

    assert "iPhone demand" in sections["Item 7"]
    assert "Revenue increased 8%" in sections["Item 7"]
    assert "TABLE OF CONTENTS" not in sections["Item 7"]

    assert "macroeconomic conditions" in sections["Item 1A"]
    assert "Management's Discussion" not in sections["Item 1A"]  # stops before Item 7's real section


def test_extract_item_sections_ignores_cross_references():
    """Regression test for the real bug found while designing this: a bare
    "Item 7" occurrence inside prose (a cross-reference, not a heading)
    must never be mistaken for the real section start."""
    sections = extract_item_sections(_html(_SAMPLE_10K_TEXT), _ITEMS)
    assert "See Item 7 of this Form 10-K" not in sections["Item 7"]
    assert not sections["Item 7"].startswith("of this Form 10-K")


def test_extract_item_sections_omits_missing_item():
    sections = extract_item_sections(_html(_SAMPLE_10K_TEXT), {**_ITEMS, "Item 99": "Nonexistent Section"})
    assert "Item 99" not in sections
    assert "Item 1A" in sections


def test_extract_item_sections_truncates_long_section():
    long_body = "x" * (MAX_SECTION_CHARS + 5000)
    text = f"Item 7.\nManagement\nItem 7.\nManagement\n{long_body}"
    sections = extract_item_sections(_html(text), {"Item 7": "Management"})
    assert len(sections["Item 7"]) == MAX_SECTION_CHARS


def test_filing_document_url_strips_dashes_and_leading_zeros():
    url = filing_document_url("0000320193", "0000320193-25-000079", "aapl-20250927.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def test_get_cik_for_ticker_resolves_from_map():
    with patch(
        "services.edgar_service._fetch_ticker_to_cik_map",
        return_value={"AAPL": "0000320193"},
    ):
        assert get_cik_for_ticker("aapl") == "0000320193"


def test_get_cik_for_ticker_returns_none_for_unknown_ticker():
    """Funds/ETFs mostly aren't SEC-registered operating companies -- an
    unresolved ticker is an honest None, not an error."""
    with patch("services.edgar_service._fetch_ticker_to_cik_map", return_value={}):
        assert get_cik_for_ticker("XLK") is None


def test_get_cik_for_ticker_returns_none_on_fetch_failure():
    with patch("services.edgar_service._fetch_ticker_to_cik_map", side_effect=RuntimeError("network down")):
        assert get_cik_for_ticker("AAPL") is None


# --- SUM-2: 8-K earnings-release discovery ---

_SAMPLE_SUBMISSIONS_RECENT = {
    "form": ["10-K", "8-K", "8-K", "8-K"],
    "accessionNumber": ["0000320193-25-000079", "0000320193-26-000018", "0001140361-26-015711", "0000320193-26-000011"],
    "filingDate": ["2025-10-31", "2026-07-30", "2026-04-20", "2026-04-30"],
    "reportDate": ["2025-09-27", "2026-07-30", "2026-04-20", "2026-04-30"],
    "primaryDocument": ["aapl-20250927.htm", "aapl-20260730.htm", "ef20071035_8k.htm", "aapl-20260430.htm"],
    "items": ["", "2.02,9.01", "5.02", "2.02,9.01"],
}


def test_get_recent_8k_filings_filters_to_earnings_item_only():
    """A 10-K and a non-earnings 8-K (item 5.02, an executive-departure
    filing -- a real item code seen on Apple's own filing history) must
    both be excluded; only item-2.02 8-Ks are earnings releases."""
    with patch("services.edgar_service._fetch_submissions_recent", return_value=_SAMPLE_SUBMISSIONS_RECENT):
        filings = get_recent_8k_filings("0000320193")

    assert len(filings) == 2
    assert {f["accession_number"] for f in filings} == {"0000320193-26-000018", "0000320193-26-000011"}
    # newest first
    assert filings[0]["filing_date"] == "2026-07-30"


def test_get_recent_8k_filings_empty_when_none_match():
    recent = {**_SAMPLE_SUBMISSIONS_RECENT, "form": ["10-K", "8-K"], "items": ["", "5.02"]}
    with patch("services.edgar_service._fetch_submissions_recent", return_value={**recent}):
        assert get_recent_8k_filings("0000320193") == []


def _index_json(names: list[str]) -> dict:
    return {"directory": {"item": [{"name": n} for n in names]}}


def test_find_exhibit_991_document_matches_real_apple_naming():
    """Verified against a real live Apple 8-K filing while designing this."""
    with patch("services.edgar_service.EDGAR_USER_AGENT", "Test test@example.com"), patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = _index_json(
            ["0000320193-26-000018-index.html", "a8-kex991q3202606272026.htm", "aapl-20260730.htm"]
        )
        mock_get.return_value.raise_for_status = lambda: None
        assert find_exhibit_991_document("0000320193", "0000320193-26-000018") == "a8-kex991q3202606272026.htm"


def test_find_exhibit_991_document_matches_real_microsoft_naming():
    """Verified against a real live Microsoft 8-K filing -- a very
    different naming convention from Apple's, both correctly matched by
    the same loose "ex99" pattern."""
    with patch("services.edgar_service.EDGAR_USER_AGENT", "Test test@example.com"), patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = _index_json(["msft-20260729.htm", "msft-ex99_1.htm", "R1.htm"])
        mock_get.return_value.raise_for_status = lambda: None
        assert find_exhibit_991_document("0000789019", "0001193125-26-323632") == "msft-ex99_1.htm"


def test_find_exhibit_991_document_prefers_991_over_992():
    with patch("services.edgar_service.EDGAR_USER_AGENT", "Test test@example.com"), patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = _index_json(["ex-99.2.htm", "ex-99.1.htm"])
        mock_get.return_value.raise_for_status = lambda: None
        assert find_exhibit_991_document("0000320193", "0000320193-26-000018") == "ex-99.1.htm"


def test_find_exhibit_991_document_returns_none_when_absent():
    with patch("services.edgar_service.EDGAR_USER_AGENT", "Test test@example.com"), patch("httpx.get") as mock_get:
        mock_get.return_value.json.return_value = _index_json(["ef20071035_8k.htm", "R1.htm"])
        mock_get.return_value.raise_for_status = lambda: None
        assert find_exhibit_991_document("0000320193", "0001140361-26-015711") is None


def test_find_exhibit_991_document_returns_none_on_fetch_failure():
    with patch("services.edgar_service.EDGAR_USER_AGENT", "Test test@example.com"), patch(
        "httpx.get", side_effect=RuntimeError("network down")
    ):
        assert find_exhibit_991_document("0000320193", "0000320193-26-000018") is None
