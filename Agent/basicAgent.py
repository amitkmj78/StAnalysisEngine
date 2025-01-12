import yfinance as yf
from datetime import datetime
info=""
def get_basic_stock_info(ticker: str) -> str:
    """
    Fetches and returns a detailed overview of a company's stock information for a given ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    str: A formatted string containing the company's stock realted information or an error message.
    {info}
    """
    try:
        # Fetch stock information
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"No data found for the ticker {ticker}."

        # Get the current date
        today_date = datetime.now().strftime('%Y-%m-%d')

        # Construct the detailed stock overview
        return f"""
        **Stock Information for {ticker} as of {today_date}:**
        
        - **Name:** {info.get('longName', 'N/A')}
        - **Sector:** {info.get('sector', 'N/A')}
        - **Industry:** {info.get('industry', 'N/A')}
        - **Current Price:** ${info.get('currentPrice', 'N/A')}
        - **Market Cap:** {info.get('marketCap', 'N/A')}
        - **Full-Time Employees:** {info.get('fullTimeEmployees', 'N/A')}
        - **Enterprise Value:** {info.get('enterpriseValue', 'N/A')}
        - **200-Day Average:** ${info.get('twoHundredDayAverage', 'N/A')}
        - **52-Week High/Low:** ${info.get('fiftyTwoWeekHigh', 'N/A')} / ${info.get('fiftyTwoWeekLow', 'N/A')}
        - **Trailing P/E:** {info.get('trailingPE', 'N/A')}
        - **Forward P/E:** {info.get('forwardPE', 'N/A')}
        - **EBITDA:** {info.get('ebitda', 'N/A')}
        - **Total Revenue:** {info.get('totalRevenue', 'N/A')}
        - **Revenue Per Share:** ${info.get('revenuePerShare', 'N/A')}
        - **Operating Cashflow:** {info.get('operatingCashflow', 'N/A')}

        **--- Highlights:**
        - Recent trading activity includes notable price fluctuations:
          - **Previous Close:** ${info.get('previousClose', 'N/A')}
          - **Day Range (Low/High):** ${info.get('regularMarketDayLow', 'N/A')} / ${info.get('regularMarketDayHigh', 'N/A')}
        - Analyst targets:
          - **Target High Price:** ${info.get('targetHighPrice', 'N/A')}
          - **Target Low Price:** ${info.get('targetLowPrice', 'N/A')}
          - **Target Mean Price:** ${info.get('targetMeanPrice', 'N/A')}

        This information reflects the most recent data available for {ticker}.
        """
    except Exception as e:
        # Handle exceptions and return a formatted error message
        return f"An error occurred while fetching data for {ticker}: {str(e)}"
