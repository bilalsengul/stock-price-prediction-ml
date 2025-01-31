import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataLoader:
    def __init__(self, symbol: str, start_date: str = None, end_date: str = None):
        """
        Initialize the StockDataLoader with a stock symbol and date range.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL' for Apple)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        try:
            logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
            stock = yf.Ticker(self.symbol)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
                
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """
        Save the stock data to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame containing the stock data
            filepath (str): Path where to save the CSV file
        """
        try:
            df.to_csv(filepath)
            logger.info(f"Data saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of StockDataLoader.
    """
    # Example usage
    symbol = 'AAPL'  # Apple stock as an example
    loader = StockDataLoader(symbol)
    
    try:
        # Fetch the data
        df = loader.fetch_data()
        
        # Save to CSV
        output_path = 'data/stock_data.csv'
        loader.save_data(df, output_path)
        
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")

if __name__ == "__main__":
    main() 