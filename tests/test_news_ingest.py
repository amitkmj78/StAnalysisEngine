from services.news_ingest_service import build_news_rows, event_for_items


def test_the_most_important_item_sets_the_event_type():
    assert event_for_items(["9.01", "2.02"]) == ("EARNINGS", "Results of operations and financial condition")
    assert event_for_items(["5.02", "8.01"])[0] == "EXECUTIVE_CHANGE"
    assert event_for_items(["4.02", "2.02"])[0] == "RESTATEMENT"


def test_a_filing_with_no_known_item_is_other_not_an_invented_type():
    assert event_for_items([]) == ("OTHER", "Form 8-K filing")
    assert event_for_items(["7.99"]) == ("OTHER", "Form 8-K filing")


def test_each_filing_becomes_one_row_with_its_own_document_link():
    filings = [
        {"accession_number": "0000320193-26-000100", "filing_date": "2026-10-01",
         "primary_document": "aapl-8k.htm", "items": ["5.02", "9.01"]},
    ]
    rows = build_news_rows(filings, cik="320193")
    assert len(rows) == 1
    row = rows[0]
    assert row["source_key"] == "0000320193-26-000100"
    assert row["title"] == "Form 8-K: Change in directors or officers"
    assert row["publisher"] == "SEC EDGAR"
    assert row["filed_on"] == "2026-10-01"
    assert row["item_codes"] == ["5.02", "9.01"]
    assert row["url"].startswith("https://www.sec.gov/") and "aapl-8k.htm" in row["url"]
