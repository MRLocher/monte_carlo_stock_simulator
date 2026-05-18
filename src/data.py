#functions used for stock data importation and cleaning.
import yfinance as yf

def download_stock_data(stock_name, start_date, end_date):
    """Download stock data for a given stock and date range and checks if the date range is within the recorded period.

    Args:
        stock_name (string): name of the stock to download data for (e.g., 'AAPL' for Apple Inc.)
        start_date (date): start date for the stock data in 'YYYY-MM-DD' format
        end_date (date): end date for the stock data in 'YYYY-MM-DD' format

    Returns:
        stock_data (pandas DataFrame): stock data for the specified date range
    """
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    #verifying the date range is within the recorded period
    if not stock_data.empty: 
        earliest_date = yf.Ticker(stock_name).history(period="max", interval="1d").index.min().date()
        if earliest_date > start_date:
            print(f"Warning: Date requested ({start_date}) is older than the stock launch date({earliest_date}). Range is automatically restricted to {earliest_date} - {end_date}.")
    return stock_data


def clean_stock_data(stock_data):
    """Clean the stock data by handling missing values and ensuring the data is sorted by date.

    Args:
        stock_data (pandas DataFrame): raw stock data

    Returns:
        clean_stock_data (pandas DataFrame): cleaned stock data
    """
    # Handle missing values
    clean_stock_data = stock_data.dropna()

    # Ensure data is sorted by date
    clean_stock_data = clean_stock_data.sort_index()

    return clean_stock_data