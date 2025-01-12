import datetime

# Get today's date
today_date = datetime.date.today()

def research(company_stock: str, user_prompt: str = None) -> str:
    """
    Collect and summarize recent news about a given stock.

    Parameters:
    - company_stock (str): The stock ticker symbol.
    - user_prompt (str): Optional custom user-defined prompt.

    Returns:
    - str: Research summary prompt.
    """
    today_date = datetime.date.today()

    # Default prompt
    default_prompt = f"""
    Collect and summarize recent news as of {today_date}, articles, press releases, and market analyses 
    related to the {company_stock} stock and its industry. Pay special attention to any significant events, 
    market sentiments, and analysts' opinions. Include upcoming events like earnings and others.
    
    expected_output: 
    A report that includes a comprehensive summary of the latest news, including today's date {today_date}, 
    any notable shifts in market sentiment, and potential impacts on the stock.
    """

    # Use custom prompt if provided, else use the default
    return user_prompt or default_prompt
