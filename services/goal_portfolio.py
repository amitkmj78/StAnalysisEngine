"""Build a starting portfolio from a goal's monthly contribution: half to one broad fund, half split across the stock
picks. Whole shares only, rounded down, so any money left over is kept as cash and reported. Pure, so it can be tested
without prices or a database."""

import math


def allocate_monthly(monthly_amount: float, fund: tuple[str, float] | None, stocks: list[tuple[str, float]],
                     fund_share: float = 0.5) -> dict:
    """fund: (ticker, price) or None. stocks: [(ticker, price)]. Returns the holdings to buy and the cash left over.
    When there are no stock picks, the stock half is held as cash rather than spread over nothing."""
    fund_budget = monthly_amount * fund_share
    stock_budget = monthly_amount - fund_budget
    holdings = []
    cash = 0.0

    if fund is None:
        cash += fund_budget
    else:
        ticker, price = fund
        shares = math.floor(fund_budget / price) if price > 0 else 0
        if shares > 0:
            holdings.append({"ticker": ticker, "shares": shares, "price": price, "role": "fund"})
        cash += fund_budget - shares * price

    if not stocks:
        cash += stock_budget
    else:
        per_stock = stock_budget / len(stocks)
        for ticker, price in stocks:
            shares = math.floor(per_stock / price) if price > 0 else 0
            if shares > 0:
                holdings.append({"ticker": ticker, "shares": shares, "price": price, "role": "stock"})
            cash += per_stock - shares * price

    return {"holdings": holdings, "cash": round(cash, 2), "invested": round(monthly_amount - cash, 2)}
