import io

import pandas as pd
import pytest

from services.positions_from_csv import compute_positions_from_trades, load_broker_activity_csv

CSV_WITH_DATES = """Activity Date,Process Date,Instrument,Description,Trans Code,Quantity,Price,Amount
1/2/2026,1/4/2026,AAPL,Apple Inc,BUY,10,150.00,$1500.00
3/15/2026,3/17/2026,AAPL,Apple Inc,BUY,5,180.00,$900.00
2/1/2026,2/3/2026,MSFT,Microsoft Corp,BUY,2,400.00,$800.00
"""

CSV_NO_DATE_COLUMN = """Instrument,Trans Code,Quantity,Price
AAPL,BUY,10,150.00
"""


def test_load_broker_activity_csv_extracts_activity_date():
    df = load_broker_activity_csv(io.BytesIO(CSV_WITH_DATES.encode()))
    assert "Date" in df.columns
    assert df["Date"].notna().all()
    aapl_dates = df[df["Ticker"] == "AAPL"]["Date"].tolist()
    assert pd.Timestamp("2026-01-02") in aapl_dates
    assert pd.Timestamp("2026-03-15") in aapl_dates


def test_load_broker_activity_csv_tolerates_missing_date_column():
    df = load_broker_activity_csv(io.BytesIO(CSV_NO_DATE_COLUMN.encode()))
    assert "Date" in df.columns
    assert df["Date"].isna().all()


def test_compute_positions_from_trades_uses_earliest_buy_as_acquired_at():
    trades = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL", "MSFT"],
            "Side": ["BUY", "BUY", "BUY"],
            "Quantity": [10.0, 5.0, 2.0],
            "Price": [150.0, 180.0, 400.0],
            "Date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-03-15"), pd.Timestamp("2026-02-01")],
        }
    )
    positions = compute_positions_from_trades(trades)
    aapl = positions[positions["Ticker"] == "AAPL"].iloc[0]
    msft = positions[positions["Ticker"] == "MSFT"].iloc[0]
    # AAPL bought twice -- Acquired_At must be the EARLIER of the two buys,
    # not the later one or the last-processed row.
    assert aapl["Acquired_At"] == pd.Timestamp("2026-01-02")
    assert msft["Acquired_At"] == pd.Timestamp("2026-02-01")


def test_compute_positions_from_trades_handles_no_date_column():
    trades = pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "Side": ["BUY"],
            "Quantity": [10.0],
            "Price": [150.0],
        }
    )
    positions = compute_positions_from_trades(trades)
    assert positions.iloc[0]["Acquired_At"] is None


def test_sell_trades_do_not_affect_acquired_at():
    trades = pd.DataFrame(
        {
            "Ticker": ["AAPL", "AAPL"],
            "Side": ["BUY", "SELL"],
            "Quantity": [10.0, 3.0],
            "Price": [150.0, 200.0],
            "Date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-06-01")],
        }
    )
    positions = compute_positions_from_trades(trades)
    assert positions.iloc[0]["Acquired_At"] == pd.Timestamp("2026-01-02")
