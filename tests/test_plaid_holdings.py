from services.plaid_holdings import holdings_to_positions


def _security(security_id, ticker_symbol):
    return {"security_id": security_id, "ticker_symbol": ticker_symbol, "name": ticker_symbol}


def _holding(account_id, security_id, quantity, cost_basis=None, institution_price=None):
    return {
        "account_id": account_id,
        "security_id": security_id,
        "quantity": quantity,
        "cost_basis": cost_basis,
        "institution_price": institution_price,
    }


def test_normal_mapping():
    response = {
        "accounts": [{"account_id": "a1"}],
        "securities": [_security("sec-aapl", "AAPL")],
        "holdings": [_holding("a1", "sec-aapl", quantity=10, cost_basis=1500.0)],
    }
    df = holdings_to_positions(response)
    assert list(df["Ticker"]) == ["AAPL"]
    assert df.iloc[0]["Shares"] == 10.0
    assert df.iloc[0]["Avg_Cost"] == 150.0  # 1500 / 10


def test_security_without_ticker_symbol_is_dropped():
    response = {
        "accounts": [{"account_id": "a1"}],
        "securities": [{"security_id": "sec-cash", "ticker_symbol": None, "name": "Cash Sweep"}],
        "holdings": [_holding("a1", "sec-cash", quantity=500.0, cost_basis=500.0)],
    }
    df = holdings_to_positions(response)
    assert df.empty
    assert list(df.columns) == ["Ticker", "Shares", "Avg_Cost"]


def test_missing_cost_basis_falls_back_to_institution_price():
    response = {
        "accounts": [{"account_id": "a1"}],
        "securities": [_security("sec-msft", "MSFT")],
        "holdings": [_holding("a1", "sec-msft", quantity=5, cost_basis=None, institution_price=410.0)],
    }
    df = holdings_to_positions(response)
    assert df.iloc[0]["Avg_Cost"] == 410.0


def test_multi_account_same_ticker_aggregated_with_weighted_average_cost():
    response = {
        "accounts": [{"account_id": "a1"}, {"account_id": "a2"}],
        "securities": [_security("sec-voo", "VOO")],
        "holdings": [
            _holding("a1", "sec-voo", quantity=10, cost_basis=4000.0),   # $400/share
            _holding("a2", "sec-voo", quantity=5, cost_basis=2100.0),    # $420/share
        ],
    }
    df = holdings_to_positions(response)
    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "VOO"
    assert df.iloc[0]["Shares"] == 15.0
    # Weighted average: (4000 + 2100) / 15 = 406.666...
    assert round(df.iloc[0]["Avg_Cost"], 2) == 406.67


def test_non_positive_quantity_dropped():
    response = {
        "accounts": [{"account_id": "a1"}],
        "securities": [_security("sec-aapl", "AAPL")],
        "holdings": [_holding("a1", "sec-aapl", quantity=0, cost_basis=0.0)],
    }
    df = holdings_to_positions(response)
    assert df.empty


def test_empty_holdings_returns_empty_dataframe_not_none():
    df = holdings_to_positions({"accounts": [], "securities": [], "holdings": []})
    assert df is not None
    assert df.empty
    assert list(df.columns) == ["Ticker", "Shares", "Avg_Cost"]
