
import datetime
today_date = datetime.date.today()

def filings_analysis(company_stock: str):
    return f"""
        Analyze the latest 10-Q and 10-K filings from EDGAR for the stock {company_stock} as of today {today_date}. 
        Focus on key sections like Management's Discussion and Analysis, financial statements, insider trading activity, 
        and any disclosed risks. Extract relevant data and insights that could influence the stock's future performance.
    
    expected_output:
        Final answer must be an expanded report that now also highlights significant findings 
        from these filings, including any red flags or positive indicators for your customer.
    """
