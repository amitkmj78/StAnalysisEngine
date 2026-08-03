import logging
from typing import Optional

import yfinance as yf

from .cache_utils import ttl_cache

logger = logging.getLogger(__name__)


@ttl_cache(maxsize=128, ttl_seconds=3600)
def get_volume_and_ownership_activity(ticker: str) -> Optional[dict]:
    """
    Trading volume plus a real read on who's been buying/selling: insider
    (officer/director) transaction counts from SEC filings, and
    institutional ("outsider") holders' reported position changes — both
    straight from yfinance, nothing modeled or inferred. Any section that
    yfinance doesn't have for this ticker comes back null rather than a
    fabricated number.
    """
    if not ticker:
        return None

    try:
        t = yf.Ticker(ticker)
    except Exception as e:
        logger.warning("Error creating Ticker for %s: %s", ticker, e)
        return None

    latest_volume = None
    avg_volume_10d = None
    try:
        hist = t.history(period="1mo")
        if not hist.empty:
            latest_volume = int(hist["Volume"].iloc[-1])
            avg_volume_10d = int(hist["Volume"].tail(10).mean())
    except Exception as e:
        logger.warning("Error fetching volume for %s: %s", ticker, e)

    insider_buys = None
    insider_sells = None
    try:
        ip = t.insider_purchases
        if ip is not None and not ip.empty:
            label_col = ip.columns[0]
            buys_row = ip[ip[label_col].str.contains("Purchases", case=False, na=False)]
            if not buys_row.empty and "Trans" in ip.columns:
                insider_buys = int(buys_row["Trans"].iloc[0])
            sells_row = ip[ip[label_col].str.contains("Sales", case=False, na=False)]
            if not sells_row.empty and "Trans" in ip.columns:
                insider_sells = int(sells_row["Trans"].iloc[0])
    except Exception as e:
        logger.warning("Error fetching insider purchases for %s: %s", ticker, e)

    institutional_increased = None
    institutional_decreased = None
    institutional_unchanged = None
    institutional_holder_count = None
    institutional_as_of = None
    try:
        ih = t.institutional_holders
        if ih is not None and not ih.empty and "pctChange" in ih.columns:
            institutional_increased = int((ih["pctChange"] > 0).sum())
            institutional_decreased = int((ih["pctChange"] < 0).sum())
            institutional_unchanged = int((ih["pctChange"] == 0).sum())
            institutional_holder_count = int(len(ih))
            if "Date Reported" in ih.columns:
                institutional_as_of = str(ih["Date Reported"].iloc[0])
    except Exception as e:
        logger.warning("Error fetching institutional holders for %s: %s", ticker, e)

    return {
        "ticker": ticker,
        "latest_volume": latest_volume,
        "avg_volume_10d": avg_volume_10d,
        "insider_buys": insider_buys,
        "insider_sells": insider_sells,
        "insider_period": "Last 6 Months",
        "institutional_increased": institutional_increased,
        "institutional_decreased": institutional_decreased,
        "institutional_unchanged": institutional_unchanged,
        "institutional_holder_count": institutional_holder_count,
        "institutional_as_of": institutional_as_of,
    }
