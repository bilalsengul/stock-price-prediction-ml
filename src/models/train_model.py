import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import logging

from src.models.lstm_model import StockPriceLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare data for training by scaling features and splitting into train/val sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame with features and target
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val, scaler)
    """
    # Separate features and target
    X = df[['open', 'high', 'low', 'volume']].values
    y = df['target'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, y_train, y_val, scaler

def main():
    # Change to project root directory
    os.chdir(project_root)
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('data/processed/test_data.csv')
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_val, y_train, y_val, scaler = prepare_data(df)
        
        # Initialize model
        logger.info("Initializing model...")
        model = StockPriceLSTM(sequence_length=1, n_features=X_train.shape[1])
        
        # Train model
        logger.info("Training model...")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=100,
            batch_size=32
        )
        
        # Save scaler
        logger.info("Saving scaler...")
        os.makedirs('models', exist_ok=True)
        np.save('models/feature_scaler.npy', {
            'scale_': scaler.scale_,
            'min_': scaler.min_,
            'data_min_': scaler.data_min_,
            'data_max_': scaler.data_max_,
            'data_range_': scaler.data_range_
        })
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 