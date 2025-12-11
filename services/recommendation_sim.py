import os
import pandas as pd
import yfinance as yf
from datetime import datetime

SIM_FILE = "recommendation_sim_log.csv"

def get_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector", "Unknown")
    except:
        return "Unknown"

def load_sim_log():
    if not os.path.exists(SIM_FILE):
        return pd.DataFrame(columns=[
            "Date", "Ticker", "Sector", "Recommendation",
            "EntryPrice", "ExitPrice", "PnL_Pct", "Status"
        ])
    return pd.read_csv(SIM_FILE)

def simulate_recommendation(ticker: str, recommendation: str):
    df = load_sim_log()
    sector = get_sector(ticker)

    if recommendation.upper() == "HOLD":
        df = df.append({
            "Date": datetime.today().strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Sector": sector,
            "Recommendation": recommendation,
            "EntryPrice": "",
            "ExitPrice": "",
            "PnL_Pct": 0.0,
            "Status": "NO TRADE"
        }, ignore_index=True)
        df.to_csv(SIM_FILE, index=False)
        return None

    hist = yf.Ticker(ticker).history(period="20d")
    if len(hist) < 12:
        return "Not enough historical data"

    entry_price = hist["Open"].iloc[-11]
    exit_price = hist["Close"].iloc[-1]

    if recommendation.upper() == "BUY":
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    df = df.append({
        "Date": datetime.today().strftime("%Y-%m-%d"),
        "Ticker": ticker,
        "Sector": sector,
        "Recommendation": recommendation,
        "EntryPrice": round(entry_price, 2),
        "ExitPrice": round(exit_price, 2),
        "PnL_Pct": round(pnl_pct, 2),
        "Status": "CLOSED"
    }, ignore_index=True)

    df.to_csv(SIM_FILE, index=False)
    return pnl_pct


def evaluate_by_sector():
    df = load_sim_log()

    if df.empty or "Sector" not in df.columns:
        return pd.DataFrame()

    df_filtered = df[df["Status"] == "CLOSED"]
    if df_filtered.empty:
        return pd.DataFrame()

    sector_stats = df_filtered.groupby("Sector").agg(
        AvgPnL_Pct=("PnL_Pct", "mean"),
        WinRate_pct=lambda x: (x > 0).mean() * 100,
        Trades=("PnL_Pct", "count")
    ).sort_values("AvgPnL_Pct", ascending=False)

    return sector_stats.reset_index()
