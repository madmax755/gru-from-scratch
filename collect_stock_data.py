import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
IEX_CLOUD_API_KEY = os.getenv('IEX_CLOUD_API_KEY')



def fetch_yahoo_finance_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data for a given symbol from Yahoo Finance.

    This function uses the yfinance library to retrieve daily stock data
    including open, high, low, close prices, and volume for the specified date range.

    Parameters:
    -----------
    symbol : str
        The stock symbol to fetch data for (e.g., 'AAPL' for Apple Inc.).
    start_date : str
        The start date for the data range in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data range in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the stock data. The index is the date,
        and columns include 'Open', 'High', 'Low', 'Close', 'Volume', and other available metrics.

    Example:
    --------
    >>> data = fetch_yahoo_finance_data('AAPL', '2023-01-01', '2023-06-30')
    >>> print(data.head())
    """
    
    try:
        data = yf.Ticker(symbol).history(start=start_date, end=end_date)
        return data
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Simple Moving Average
    # df['SMA_5'] = df['Close'].rolling(window=5).mean()
    # df['SMA_10'] = df['Close'].rolling(window=10).mean()
    # df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Convert absolute values to returns
    df['Return'] = df['Close'].pct_change()
    df['Volume_Return'] = df['Volume'].pct_change()
    df['Day_Volatility'] = (df['High'] - df['Low'])/df['Close']

    
    # Relative Strength Index
    delta = df['Return'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def select_features(df, features):
    return df[features]

if __name__ == "__main__":
    # symbol = "AAPL"
    # start_date = "2020-01-01"
    # end_date = "2024-10-27"
    
    # data = compute_indicators(fetch_yahoo_finance_data(symbol, start_date, end_date)).dropna()
    # data = select_features(data, ['Return', 'Volume_Return', 'Day_Volatility', 'RSI', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower'])
    # data.to_csv(f"GRU/stock_data/{symbol}_data.csv")

    df = pd.read_csv("GRU/stock_data/AAPL_data.csv")
    df['Return'] = np.log(1+df['Return'])
    df['Volume_Return'] = np.log(1+df['Volume_Return'])
    df['Day_Volatility'] = np.log(1+df['Day_Volatility'])
    df['RSI_normalised'] = (df['RSI'] - df['RSI'].mean())/df['RSI'].std()
    df['MACD_normalised'] = (df['MACD'] - df['MACD'].mean())/df['MACD'].std()
    df.drop(columns=['RSI', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Date'], inplace=True)
    df.to_csv("GRU/stock_data/AAPL_data_log.csv")