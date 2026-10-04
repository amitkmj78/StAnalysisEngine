import pytest

from services.trade_impact_service import apply_trade, compare, concentration, portfolio_score, sector_weights


def _holdings():
    return [
        {"ticker": "AAA", "shares": 10, "market_value": 600.0, "sector": "Technology", "short_score": 80.0},
        {"ticker": "BBB", "shares": 20, "market_value": 300.0, "sector": "Energy", "short_score": 60.0},
        {"ticker": "CCC", "shares": 5, "market_value": 100.0, "sector": None, "short_score": None},
    ]


def test_concentration_is_share_of_total_value():
    c = concentration(_holdings())  # total 1000
    assert c["largest_position_pct"] == pytest.approx(60.0)
    assert c["top5_pct"] == pytest.approx(100.0)


def test_sector_weights_keep_unclassified_value_in_the_denominator():
    w = sector_weights(_holdings())
    assert w == {"Technology": 60.0, "Energy": 30.0}  # CCC's 10% is unclassified, not dropped from the total


def test_portfolio_score_is_value_weighted_and_reports_coverage():
    s = portfolio_score(_holdings())
    # (600*80 + 300*60) / 900 = 73.33; coverage is 900 of 1000
    assert s["score"] == pytest.approx(73.33, abs=0.01)
    assert s["coverage_pct"] == pytest.approx(90.0)


def test_buying_a_new_ticker_adds_a_holding_at_the_given_price():
    after = apply_trade(_holdings(), "DDD", "buy", 4, 50.0, "Financials", 70.0)
    d = next(h for h in after if h["ticker"] == "DDD")
    assert d["market_value"] == pytest.approx(200.0)
    assert concentration(after)["holdings"] == 4


def test_selling_all_removes_the_holding():
    after = apply_trade(_holdings(), "BBB", "sell", 20, 15.0, None, None)
    assert [h["ticker"] for h in after] == ["AAA", "CCC"]


def test_selling_more_than_held_is_rejected():
    with pytest.raises(ValueError, match="cannot sell"):
        apply_trade(_holdings(), "AAA", "sell", 11, 60.0, None, None)


def test_compare_reports_changes_and_sector_rows():
    before = _holdings()
    after = apply_trade(before, "BBB", "buy", 20, 15.0, "Energy", 60.0)  # BBB market value 300 -> 600
    result = compare(before, after, beta_before=1.1, beta_after=1.2)
    assert result["changes"]["largest_position_pct"] == pytest.approx(
        result["after"]["concentration"]["largest_position_pct"] - 60.0, abs=0.01
    )
    assert result["changes"]["beta"] == pytest.approx(0.1)
    energy = next(s for s in result["sectors"] if s["sector"] == "Energy")
    assert energy["before_pct"] == pytest.approx(30.0)
    assert energy["after_pct"] == pytest.approx(600 / 1300 * 100, abs=0.01)  # 600 of 1300 total
