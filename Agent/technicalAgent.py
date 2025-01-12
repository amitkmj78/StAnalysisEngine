import pandas as pd
import yfinance as yf


# Utility functions
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal



# def get_technical_analysis(ticker: str, period: str = "1y") -> pd.DataFrame:
#     """
#     Fetches technical analysis for a given stock ticker over a specified period.

#     Parameters:
#     ticker (str): The stock ticker symbol.
#     period (str): The period for historical data (e.g., "1y", "6mo", "3mo"). Default is "1y".

#     Returns:
#     pd.DataFrame: A DataFrame containing various technical indicators and their values.
#     """
#     try:
#         # Fetch stock historical data
#         stock = yf.Ticker(ticker)
#         history = stock.history(period=period)

#         if history.empty:
#             return pd.DataFrame({"Error": ["No historical data available"]})

#         # Calculate technical indicators
#         history['SMA_20'] = history['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
#         history['SMA_50'] = history['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
#         history['SMA_200'] = history['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average
#         history['EMA_20'] = history['Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average
#         history['RSI'] = calculate_rsi(history['Close'])  # Relative Strength Index (14-day)
#         history['Bollinger_Upper'] = history['SMA_20'] + (history['Close'].rolling(window=20).std() * 2)  # Bollinger Bands
#         history['Bollinger_Lower'] = history['SMA_20'] - (history['Close'].rolling(window=20).std() * 2)

#         # Get the latest values
#         latest = history.iloc[-1]

#         # Return a DataFrame with key indicators
#         return pd.DataFrame({
#             'Indicator': ['Current Price', '20-day SMA', '50-day SMA', '200-day SMA', '20-day EMA', 'RSI (14-day)', 'Bollinger Upper', 'Bollinger Lower'],
#             'Value': [
#                 f"${latest['Close']:.2f}",
#                 f"${latest['SMA_20']:.2f}",
#                 f"${latest['SMA_50']:.2f}",
#                 f"${latest['SMA_200']:.2f}",
#                 f"${latest['EMA_20']:.2f}",
#                 f"{latest['RSI']:.2f}",
#                 f"${latest['Bollinger_Upper']:.2f}",
#                 f"${latest['Bollinger_Lower']:.2f}"
#             ]
#         })

#     except Exception as e:
#         # Handle exceptions and return an error DataFrame
#         return pd.DataFrame({"Error": [f"An error occurred: {str(e)}"]})


# Define Tool 2: Stock Technical Analysis
def get_technical_analysis(ticker: str, period: str = "1y") -> str:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)

        if history.empty:
            return f"No historical data available for {ticker}."

        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        history['SMA_50'] = history['Close'].rolling(window=50).mean()

        return f"""
        Technical analysis for {ticker} over {period}:
        - Current Price: {history['Close'].iloc[-1]}
        - 20-day SMA: {history['SMA_20'].iloc[-1]}
        - 50-day SMA: {history['SMA_50'].iloc[-1]}
        """
    except Exception as e:
        return f"Error performing technical analysis for {ticker}: {str(e)}"
