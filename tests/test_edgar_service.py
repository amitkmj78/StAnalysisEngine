from unittest.mock import patch

import pytest

from services.edgar_service import (
    MAX_SECTION_CHARS,
    extract_item_sections,
    filing_document_url,
    get_cik_for_ticker,
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
