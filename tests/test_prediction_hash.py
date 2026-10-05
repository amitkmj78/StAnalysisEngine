from services.prediction_hash_service import chain_hash, day_content_hash, record_hash, verify_chain


def _row(ticker="AAPL", short="Buy", score=62.4):
    return {
        "ticker": ticker, "universe_id": "All", "as_of_date": "2026-10-01",
        "short_score": score, "short_signal": short, "long_score": 61.6, "long_signal": "Hold",
        "sector_key": "Information Technology", "regime": "Neutral",
    }


def test_the_same_predictions_always_give_the_same_hash_in_any_order():
    rows = [_row("AAPL"), _row("MSFT", score=57.3)]
    assert day_content_hash(rows) == day_content_hash(list(reversed(rows)))
    assert record_hash(rows[0]) == record_hash(dict(rows[0]))


def test_changing_any_field_changes_the_hash():
    base = _row()
    assert record_hash(base) != record_hash({**base, "short_signal": "Trim"})
    assert record_hash(base) != record_hash({**base, "short_score": 62.5})


def test_removing_a_prediction_changes_the_day_hash():
    rows = [_row("AAPL"), _row("MSFT")]
    assert day_content_hash(rows) != day_content_hash(rows[:1])


def _publish(days):
    entries, previous = [], None
    for day, rows in days:
        content = day_content_hash(rows)
        chained = chain_hash(previous, day, len(rows), content)
        entries.append({"day": day, "record_count": len(rows), "content_hash": content, "chain_hash": chained})
        previous = chained
    return entries


def test_an_unchanged_record_verifies_and_an_altered_past_day_is_caught():
    days = [("2026-10-01", [_row("AAPL")]), ("2026-10-02", [_row("MSFT", score=57.3)])]
    entries = _publish(days)
    stored = {day: rows for day, rows in days}
    assert verify_chain(entries, stored) == {"ok": True, "days_checked": 2}

    tampered = {**stored, "2026-10-01": [_row("AAPL", short="Trim")]}
    result = verify_chain(entries, tampered)
    assert result["ok"] is False
    assert result["first_failing_day"] == "2026-10-01"


def test_a_missing_prediction_is_caught_even_if_the_later_days_are_intact():
    days = [("2026-10-01", [_row("AAPL"), _row("MSFT")]), ("2026-10-02", [_row("NVDA")])]
    entries = _publish(days)
    stored = {"2026-10-01": [_row("AAPL")], "2026-10-02": [_row("NVDA")]}
    assert verify_chain(entries, stored)["ok"] is False
