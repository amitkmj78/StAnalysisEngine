import pandas as pd

def build_manual_positions(
    names: list[str],
    tickers: list[str],
    shares: list[float],
    current_prices: list[float],
    avg_costs: list[float],
    total_returns: list[float] | None = None,
    acquired_dates: list[str | None] | None = None,
) -> pd.DataFrame:
    """
    Create a standardized holdings DataFrame from manual user input.
    Output columns MATCH what your strategy engine expects:

    - Ticker
    - Name
    - Shares
    - Current_Price
    - Avg_Cost
    - Unrealized_PnL_%
    - Acquired_At (the date the user says they bought it, when given --
      None otherwise, which _save_and_respond then defaults to today)
    """
    data = []

    for i in range(len(tickers)):
        if not tickers[i]:
            continue

        shr = float(shares[i]) if shares[i] else 0
        price = float(current_prices[i]) if current_prices[i] else 0
        cost = float(avg_costs[i]) if avg_costs[i] else 0

        if shr <= 0:
            continue

        # compute PnL %
        if total_returns and total_returns[i] not in (None, ""):
            pnl_pct = float(total_returns[i])
        else:
            pnl_pct = ((price - cost) / cost * 100) if cost > 0 else 0

        acquired_at = acquired_dates[i] if acquired_dates and i < len(acquired_dates) else None

        data.append(
            {
                "Ticker": tickers[i].upper().strip(),
                "Name": names[i].strip() if names[i] else tickers[i].upper(),
                "Shares": shr,
                "Avg_Cost": cost,
                "Current_Price": price,
                "Unrealized_PnL_%": pnl_pct,
                "Acquired_At": acquired_at or None,
            }
        )

    return pd.DataFrame(data)
