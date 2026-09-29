from services.factor_narrative_service import (
    _ordinal,
    earnings_revisions_sentence,
    earnings_surprise_sentence,
    growth_sentence,
    low_vol_sentence,
    momentum_sentence,
    quality_sentence,
    reversal_sentence,
    value_sentence,
)


def test_ordinal_suffixes():
    # Regression: every sentence used to hardcode "th", so "53th
    # percentile" / "21th percentile" / "1th percentile" were real,
    # visible grammar bugs.
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(22) == "22nd"
    assert _ordinal(23) == "23rd"
    assert _ordinal(53) == "53rd"
    assert _ordinal(100) == "100th"
    assert _ordinal(111) == "111th"


def test_momentum_sentence_uses_correct_ordinal_suffix():
    assert "53rd percentile" in momentum_sentence(6.3, 30, 53.0)


def test_momentum_sentence_with_real_numbers():
    text = momentum_sentence(12.3, 30, 80.0)
    assert "up 12.3%" in text
    assert "30 trading days" in text
    assert "80th percentile" in text


def test_momentum_sentence_negative_return_says_down():
    text = momentum_sentence(-5.0, 10, 20.0)
    assert "down 5.0%" in text


def test_momentum_sentence_none_degrades_gracefully():
    text = momentum_sentence(None, 30, None)
    assert "Not enough" in text


def test_reversal_sentence_labels_oversold():
    text = reversal_sentence(25.0, 90.0)
    assert "oversold" in text
    assert "90th percentile" in text


def test_reversal_sentence_labels_overbought():
    text = reversal_sentence(75.0, 10.0)
    assert "overbought" in text


def test_reversal_sentence_none_degrades_gracefully():
    assert "Not enough" in reversal_sentence(None, None)


def test_value_sentence_with_real_numbers():
    text = value_sentence(18.5, 70.0)
    assert "18.5" in text
    assert "70th percentile" in text


def test_value_sentence_none_degrades_gracefully():
    assert "No forward P/E" in value_sentence(None, None)


def test_growth_sentence_with_both_numbers():
    text = growth_sentence(6.0, 12.0, 65.0)
    assert "revenue growth of 6.0%" in text
    assert "earnings growth of 12.0%" in text
    assert "65th percentile" in text


def test_growth_sentence_with_only_one_number():
    text = growth_sentence(None, 12.0, 65.0)
    assert "earnings growth of 12.0%" in text
    assert "revenue" not in text


def test_growth_sentence_none_percentile_degrades_gracefully():
    assert "Not enough" in growth_sentence(6.0, 12.0, None)


def test_low_vol_sentence_with_real_numbers():
    text = low_vol_sentence(18.0, 85.0)
    assert "18.0%" in text
    assert "85th percentile" in text


def test_low_vol_sentence_none_degrades_gracefully():
    assert "Not enough" in low_vol_sentence(None, None)


def test_quality_sentence_with_real_numbers():
    text = quality_sentence(22.5, 75.0)
    assert "22.5%" in text
    assert "75th percentile" in text


def test_quality_sentence_none_degrades_gracefully():
    assert "Not enough" in quality_sentence(None, None)


def test_earnings_surprise_sentence_says_beat():
    text = earnings_surprise_sentence(6.7, 80.0)
    assert "beat estimates by 6.7%" in text
    assert "80th percentile" in text


def test_earnings_surprise_sentence_says_missed():
    text = earnings_surprise_sentence(-3.2, 15.0)
    assert "missed estimates by 3.2%" in text


def test_earnings_surprise_sentence_none_degrades_gracefully():
    assert "No recent earnings surprise" in earnings_surprise_sentence(None, None)


def test_earnings_revisions_sentence_says_raised():
    text = earnings_revisions_sentence(2.1, 85.0)
    assert "raised 2.1%" in text
    assert "85th percentile" in text


def test_earnings_revisions_sentence_says_cut():
    text = earnings_revisions_sentence(-4.4, 10.0)
    assert "cut 4.4%" in text


def test_earnings_revisions_sentence_none_degrades_gracefully():
    assert "Not enough analyst-estimate history" in earnings_revisions_sentence(None, None)
