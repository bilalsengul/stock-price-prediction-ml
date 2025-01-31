import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        """
        Initialize the TechnicalIndicators class with default parameters.
        """
        pass
        
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        try:
            df = df.copy()
            
            # Add individual indicators
            df = self.add_moving_averages(df)
            df = self.add_rsi(df)
            df = self.add_macd(df)
            df = self.add_bollinger_bands(df)
            
            # Remove any NaN values
            df = df.fillna(method='bfill')
            
            logger.info("Successfully added all technical indicators")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Simple and Exponential Moving Averages.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with moving averages
        """
        try:
            # Add Simple Moving Averages
            sma_20 = SMAIndicator(close=df['Close'], window=20)
            sma_50 = SMAIndicator(close=df['Close'], window=50)
            sma_200 = SMAIndicator(close=df['Close'], window=200)
            
            df['SMA20'] = sma_20.sma_indicator()
            df['SMA50'] = sma_50.sma_indicator()
            df['SMA200'] = sma_200.sma_indicator()
            
            # Add Exponential Moving Averages
            ema_12 = EMAIndicator(close=df['Close'], window=12)
            ema_26 = EMAIndicator(close=df['Close'], window=26)
            
            df['EMA12'] = ema_12.ema_indicator()
            df['EMA26'] = ema_26.ema_indicator()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")
            raise
            
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with RSI
        """
        try:
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()
            return df
            
        except Exception as e:
            logger.error(f"Error adding RSI: {str(e)}")
            raise
            
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with MACD
        """
        try:
            macd = MACD(close=df['Close'])
            
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding MACD: {str(e)}")
            raise
            
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands
        """
        try:
            bollinger = BollingerBands(close=df['Close'])
            
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Lower'] = bollinger.bollinger_lband()
            
            # Add Bollinger Band width and percentage
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Percentage'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding Bollinger Bands: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of TechnicalIndicators.
    """
    try:
        # Load example data
        df = pd.read_csv('data/stock_data.csv', index_col=0)
        
        # Initialize technical indicators
        indicators = TechnicalIndicators()
        
        # Add all indicators
        df_with_indicators = indicators.add_all_indicators(df)
        
        # Save the enhanced dataset
        df_with_indicators.to_csv('data/stock_data_with_indicators.csv')
        
        logger.info("Technical indicators added successfully")
        
    except Exception as e:
        logger.error(f"Failed to add technical indicators: {str(e)}")

if __name__ == "__main__":
    main() 