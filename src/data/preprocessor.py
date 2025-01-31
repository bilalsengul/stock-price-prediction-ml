import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataPreprocessor with necessary scalers and parameters.
        """
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.sequence_length = 60  # Number of time steps to look back
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Create copy to avoid modifying original data
            df = df.copy()
            
            # Handle missing values
            df = df.fillna(method='ffill')  # Forward fill
            df = df.fillna(method='bfill')  # Backward fill for any remaining NaNs
            
            # Remove outliers using IQR method
            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info("Data preparation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def scale_data(self, df: pd.DataFrame) -> tuple:
        """
        Scale the features using MinMaxScaler.
        
        Args:
            df (pd.DataFrame): Preprocessed stock data
            
        Returns:
            tuple: Scaled price data and volume data
        """
        try:
            # Scale prices (Open, High, Low, Close)
            price_data = df[['Open', 'High', 'Low', 'Close']].values
            scaled_prices = self.price_scaler.fit_transform(price_data)
            
            # Scale volume separately
            volume_data = df[['Volume']].values
            scaled_volume = self.volume_scaler.fit_transform(volume_data)
            
            logger.info("Data scaling completed successfully")
            return scaled_prices, scaled_volume
            
        except Exception as e:
            logger.error(f"Error in data scaling: {str(e)}")
            raise
            
    def create_sequences(self, price_data: np.ndarray, volume_data: np.ndarray) -> tuple:
        """
        Create sequences for time series prediction.
        
        Args:
            price_data (np.ndarray): Scaled price data
            volume_data (np.ndarray): Scaled volume data
            
        Returns:
            tuple: X (features) and y (target) arrays for training
        """
        try:
            X, y = [], []
            
            # Combine price and volume data
            combined_data = np.hstack((price_data, volume_data))
            
            for i in range(len(combined_data) - self.sequence_length):
                # Create sequence of past data points
                sequence = combined_data[i:(i + self.sequence_length)]
                # Target is the next day's closing price
                target = price_data[i + self.sequence_length, 3]  # Close price
                
                X.append(sequence)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created {len(X)} sequences for training")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in sequence creation: {str(e)}")
            raise
            
    def inverse_transform_prices(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Convert scaled prices back to original scale.
        
        Args:
            scaled_data (np.ndarray): Scaled price data
            
        Returns:
            np.ndarray: Original scale price data
        """
        try:
            # Reshape if necessary
            if len(scaled_data.shape) == 1:
                scaled_data = scaled_data.reshape(-1, 1)
            
            # Create dummy data for other price columns (Open, High, Low)
            dummy_data = np.zeros((len(scaled_data), 4))
            dummy_data[:, 3] = scaled_data.flatten()  # Put the closing prices in the last column
            
            # Inverse transform
            original_scale = self.price_scaler.inverse_transform(dummy_data)[:, 3]
            
            return original_scale
            
        except Exception as e:
            logger.error(f"Error in inverse transformation: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of DataPreprocessor.
    """
    try:
        # Load example data
        df = pd.read_csv('data/stock_data.csv', index_col=0)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Prepare and scale data
        df_cleaned = preprocessor.prepare_data(df)
        scaled_prices, scaled_volume = preprocessor.scale_data(df_cleaned)
        
        # Create sequences
        X, y = preprocessor.create_sequences(scaled_prices, scaled_volume)
        
        logger.info(f"Final dataset shape - X: {X.shape}, y: {y.shape}")
        
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")

if __name__ == "__main__":
    main() 