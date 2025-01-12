
import datetime
today_date = datetime.date.today()

def research(company_stock: str):
    return f"""
    Collect and summarize recent news as of today date {today_date} articles, press releases, and market analyses 
    related to the {{company_stock}} stock and its industry. Pay special attention to any significant events, 
    market sentiments, and analysts' opinions. Also include upcoming events like earnings and others. 
    expected_output: 
     A report that includes a comprehensive summary of the latest news, including today's date ({today_date}), 
    any notable shifts in market sentiment, and potential impacts on the stock. Also, make sure to return the stock 
    ticker as {company_stock}. Make sure to use the most recent data possible.   

    """
