import yfinance as yf
import datetime
from statistics import mean
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup

today_date = datetime.date.today()

# Global LLM
llm_financial = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# -------------------------------------------------------------
# SAFE GET
# -------------------------------------------------------------
def safe(info, key, default="N/A"):
    try:
        value = info.get(key, default)
        return value if value not in (None, "", "None") else default
    except:
        return default


# -------------------------------------------------------------
# FETCH FINANCIAL DATA
# -------------------------------------------------------------
def fetch_financials(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return None


# -------------------------------------------------------------
# ROE + DUPONT ANALYSIS
# -------------------------------------------------------------
def calculate_roe_and_dupont(info, ticker_obj):
    try:
        financials = ticker_obj.financials
        balance = ticker_obj.balance_sheet
        if financials is None or balance is None:
            raise ValueError("Missing statements")

        latest_col = financials.columns[0]

        net_income = financials.loc["Net Income", latest_col]
        revenue = financials.loc["Total Revenue", latest_col]
        assets = balance.loc["Total Assets", latest_col]
        equity = balance.loc["Stockholders Equity", latest_col]

        if any(v in (None, 0) for v in [net_income, revenue, assets, equity]):
            raise ValueError("Invalid ROE components")

        pm = net_income / revenue
        at = revenue / assets
        em = assets / equity
        roe = pm * at * em

        return {
            "ROE": round(roe, 4),
            "ProfitMargin": round(pm, 4),
            "AssetTurnover": round(at, 4),
            "EquityMultiplier": round(em, 4)
        }

    except:
        return {
            "ROE": "N/A", "ProfitMargin": "N/A",
            "AssetTurnover": "N/A", "EquityMultiplier": "N/A"
        }


# -------------------------------------------------------------
# HEALTH SCORE CALC
# -------------------------------------------------------------
def compute_health_score(info):
    score = []

    pe = safe(info, "trailingPE")
    score.append(30 if isinstance(pe, (int, float)) and pe < 25 else 15)

    gm = safe(info, "grossMargins")
    score.append(gm * 30 if isinstance(gm, (int, float)) else 10)

    dte = safe(info, "debtToEquity")
    score.append(30 if isinstance(dte, (int, float)) and dte < 100 else 15)

    fcf = safe(info, "freeCashflow")
    score.append(30 if isinstance(fcf, (int, float)) and fcf > 0 else 10)

    growth = safe(info, "revenueGrowth")
    score.append(30 if isinstance(growth, (int, float)) and growth > 0 else 10)

    return round(mean(score), 2)


# -------------------------------------------------------------
# SIMPLE TRAFFIC-LIGHT VERDICT
# -------------------------------------------------------------
def simple_verdict(score, info, dupont):
    growth = info.get("revenueGrowth")
    dte = info.get("debtToEquity")
    roe = dupont.get("ROE")

    roe_num = roe if isinstance(roe, (int, float)) else None

    good = []
    risk = []

    # -------------------------
    # POSITIVE SIGNALS
    # -------------------------
    if isinstance(growth, (int, float)) and growth > 0.05:
        good.append("Revenue is growing at a healthy rate.")

    if roe_num and roe_num >= 0.10:
        good.append("ROE is healthy, meaning the company uses capital efficiently.")

    if isinstance(dte, (int, float)) and dte < 200:
        good.append("Debt level is within a normal range for the industry.")

    if score >= 70:
        good.append("Overall financial score is strong.")

    # -------------------------
    # RISK SIGNALS
    # -------------------------
    if isinstance(growth, (int, float)) and growth < 0:
        risk.append("Revenue growth is negative.")

    if roe_num and roe_num < 0.05:
        risk.append("ROE is very low, indicating poor capital efficiency.")

    if isinstance(dte, (int, float)) and dte > 400:
        risk.append("Debt level is very high compared to equity.")

    if score < 55:
        risk.append("Financial score is on the weak side.")

    # -------------------------
    # FINAL LABEL LOGIC
    # -------------------------

    # ----- GREEN: strong overall, no major red flags -----
    if score >= 70 and (roe_num is None or roe_num >= 0.10) and (not isinstance(dte, (int, float)) or dte < 250):
        label = "🟢 GREEN"
        summary = "The company appears financially solid with generally healthy metrics."

    # ----- RED: only when BAD across multiple dimensions -----
    elif (score < 50) or \
         (isinstance(dte, (int, float)) and dte > 600) or \
         (isinstance(growth, (int, float)) and growth < -0.05):
        label = "🔴 RED"
        summary = "Multiple key financial metrics indicate elevated risk."

    # ----- YELLOW: everything in between -----
    else:
        label = "🟡 YELLOW"
        summary = "This company shows a mix of strengths and weaknesses."

    return {
        "label": label,
        "summary": summary,
        "good": good,
        "risk": risk,
    }


# -------------------------------------------------------------
# UNBIASED AI INTERPRETATION
# -------------------------------------------------------------
def ai_top_interpretation(text):
    """
    A purely objective, neutral AI interpretation.
    No opinions, no recommendations, no bias.
    Only clarifies the numbers.
    """
    try:
        res = llm_financial.invoke(
            "Provide an unbiased, strictly factual interpretation of the financial data below. "
            "Do NOT give investment advice, do NOT recommend buy/sell, do NOT express opinion. "
            "Simply summarize what the numbers objectively indicate: trends, strengths, weaknesses, and risks.\n\n"
            f"{text}"
        )
        return res.content
    except:
        return "AI interpretation unavailable."


# -------------------------------------------------------------
# PEER DETECTION
# -------------------------------------------------------------
def peer_compare(ticker):
    # Simple static universe
    universe = [
        "AAPL","MSFT","AMZN","GOOGL","META",
        "NVDA","NFLX","INTC","AMD","TSLA",
        "AVGO","CSCO","ADBE","CRM","ORCL"
    ]

    info = yf.Ticker(ticker).info
    sector = info.get("sector")
    industry = info.get("industry")

    peers = {}
    for sym in universe:
        try:
            pinfo = yf.Ticker(sym).info
            if pinfo.get("sector") == sector and pinfo.get("industry") == industry:
                peers[sym] = {
                    "MarketCap": pinfo.get("marketCap"),
                    "PE": pinfo.get("trailingPE"),
                    "GM": pinfo.get("grossMargins"),
                    "RevGrowth": pinfo.get("revenueGrowth")
                }
        except:
            continue

    return peers if peers else {"error": "No peers found"}


# -------------------------------------------------------------
# MAIN ENGINE
# -------------------------------------------------------------
def financial_analysis(ticker: str) -> str:

    info = fetch_financials(ticker)
    if not info:
        return f"No financial data available for {ticker}."

    extracted = {
        "Current Price": safe(info, "currentPrice"),
        "Market Cap": safe(info, "marketCap"),
        "Forward P/E": safe(info, "forwardPE"),
        "Trailing P/E": safe(info, "trailingPE"),
        "PEG Ratio": safe(info, "pegRatio"),
        "Revenue Growth": safe(info, "revenueGrowth"),
        "Earnings Growth": safe(info, "earningsGrowth"),
        "Gross Margins": safe(info, "grossMargins"),
        "Operating Margins": safe(info, "operatingMargins"),
        "Debt-to-Equity": safe(info, "debtToEquity"),
        "Free Cash Flow": safe(info, "freeCashflow"),
        "Operating Cash Flow": safe(info, "operatingCashflow"),
        "Current Ratio": safe(info, "currentRatio"),
    }

    score = compute_health_score(info)
    dupont = calculate_roe_and_dupont(info, yf.Ticker(ticker))
    peers = peer_compare(ticker)
    verdict = simple_verdict(score, info, dupont)

    # ========== BUILD THE REPORT ==========
    report = f"""
FINANCIAL ANALYSIS — {ticker}
Date: {today_date}


TRAFFIC-LIGHT RATING: **{verdict['label']}**
Summary: {verdict['summary']}

-------------------------
STRENGTHS
-------------------------
"""
    if verdict["good"]:
        for g in verdict["good"]:
            report += f"- {g}\n"
    else:
        report += "- No major strengths detected.\n"

    report += """
-------------------------
RISKS
-------------------------
"""
    if verdict["risk"]:
        for r in verdict["risk"]:
            report += f"- {r}\n"
    else:
        report += "- No major risks detected.\n"

    report += """
-------------------------
KEY FUNDAMENTALS
-------------------------
"""
    for k, v in extracted.items():
        report += f"• {k}: {v}\n"

    report += f"\nFinancial Health Score: **{score}/100**\n"

    report += """
-------------------------
ROE & DUPONT ANALYSIS
-------------------------
"""
    for k, v in dupont.items():
        report += f"• {k}: {v}\n"

    report += """
-------------------------
PEER COMPARISON
-------------------------
"""
    if "error" in peers:
        report += peers["error"] + "\n"
    else:
        for sym, data in peers.items():
            report += f"{sym}:\n"
            for k, v in data.items():
                report += f"  - {k}: {v}\n"

    report += """
-------------------------
TOP AI INTERPRETATION (UNBIASED)
-------------------------
"""
    report += ai_top_interpretation(report)

    return report
