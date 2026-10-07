from services.goal_portfolio import allocate_monthly


def test_half_goes_to_the_fund_and_half_to_the_stocks_in_whole_shares():
    result = allocate_monthly(1000, ("VTI", 250.0), [("AAA", 100.0), ("BBB", 50.0)])
    shares = {h["ticker"]: h["shares"] for h in result["holdings"]}
    assert shares == {"VTI": 2, "AAA": 2, "BBB": 5}
    assert result["cash"] == 50.0  # AAA leaves $50 after 2 whole shares


def test_money_that_cannot_buy_a_whole_share_is_kept_as_cash():
    result = allocate_monthly(1000, ("VTI", 300.0), [("AAA", 400.0)])
    assert sum(h["shares"] * h["price"] for h in result["holdings"]) + result["cash"] == 1000


def test_with_no_stock_picks_the_stock_half_is_cash_and_says_so():
    result = allocate_monthly(1000, ("VTI", 100.0), [])
    assert all(h["role"] == "fund" for h in result["holdings"])
    assert result["cash"] == 500.0


def test_with_no_fund_the_fund_half_is_cash():
    result = allocate_monthly(1000, None, [("AAA", 100.0)])
    assert result["cash"] == 500.0
