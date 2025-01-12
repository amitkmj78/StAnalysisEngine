import datetime
today_date = datetime.date.today()

def recommend(company_stock: str):
    
    return f"""
    **Investment Recommendation Report**  
    **Stock:** {company_stock}  
    **Date:** {today_date}  

    ---  

    **Executive Summary:**  
    This report synthesizes analyses provided by the Financial Analyst and the Research Analyst. It integrates insights into the financial health, market sentiment, and qualitative data extracted from EDGAR filings.  

    ---  

    **Section 1: Financial Analysis**  
    - **Key Metrics:** (Include PE ratio, EPS, revenue growth, profit margins, etc.)  
    - **Balance Sheet Health:** (Assess liquidity, debt levels, and asset quality.)  
    - **Cash Flow Trends:** (Highlight operating, investing, and financing cash flows.)  

    **Section 2: Market Sentiment**  
    - **Recent Stock Performance:** (Evaluate price movements, volatility, and trends.)  
    - **Analyst Ratings:** (Summarize ratings, price targets, and consensus opinions.)  
    - **News Sentiment:** (Include notable headlines and sentiment analysis.)  

    **Section 3: Qualitative Insights from EDGAR Filings**  
    - **Key Disclosures:** (Discuss management commentary, risks, and strategies.)  
    - **Material Changes:** (Identify significant changes in operations or outlook.)  
    - **Other Highlights:** (Extract any unique insights or observations.)  

    **Section 4: Insider Trading Activity**  
    - **Recent Transactions:** (Summarize insider buying/selling activity.)  
    - **Implications:** (Discuss whether this reflects confidence or caution.)  

    **Section 5: Upcoming Events**  
    - **Earnings Report Date:** (Include relevant dates and anticipated impacts.)  
    - **Dividends/Buybacks:** (Note any announced plans.)  
    - **Corporate Events:** (Highlight conferences, product launches, etc.)  

    ---  

    **Recommendation:**  
    Based on the analysis above, our recommendation for {company_stock} is:  
    - **Investment Stance:** (e.g., Buy, Hold, or Sell)  
    - **Rationale:** (Summarize key factors supporting this stance.)  
    - **Strategy:** (Provide actionable advice, such as entry/exit points, long-term expectations, and risk management tips.)  

    ---  

    **Supporting Evidence:**  
    - Include relevant charts, graphs, or tables to enhance understanding.  
    - Ensure the report is professional and visually appealing for the customer.  

    ---  

    **Note:**  
    This recommendation is based on the data available as of {today_date}. Market conditions may change, and it is advised to reassess periodically.  

    """

