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
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm

from src.models.lstm_model import StockPriceLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            self.status = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.counter >= self.patience:
                self.status = f'EarlyStopping triggered'
                return True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0
            self.status = f'EarlyStopping counter: {self.counter}'
        return False

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    # Moving averages
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Log transform volume
    df['volume'] = np.log1p(df['volume'])
    
    return df

def create_sequences(X, y, seq_length=10):
    """Create sequences for time series prediction."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def add_noise(data, noise_level=0.01):
    """Add random noise to the data for augmentation."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def prepare_price_features(df):
    """Prepare price-based features using percentage changes and momentum."""
    # Calculate percentage changes
    df['return_1d'] = df['close'].pct_change()
    df['high_low_pct'] = (df['high'] - df['low']) / df['low']
    df['open_close_pct'] = (df['close'] - df['open']) / df['open']
    
    # Price momentum features
    df['momentum_1d'] = df['close'].pct_change()
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    
    # Volatility features
    df['volatility_5d'] = df['return_1d'].rolling(window=5).std()
    df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
    
    # Moving average crossovers
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma_crossover'] = (df['ma5'] - df['ma10']) / df['ma10']
    
    # Target variable (next day's return)
    df['target'] = df['close'].pct_change().shift(-1)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def prepare_data(df: pd.DataFrame, seq_length: int = 10) -> tuple:
    """Prepare data for training with augmentation."""
    # Prepare features
    df = prepare_price_features(df)
    
    # Select features
    feature_columns = [
        'return_1d', 'high_low_pct', 'open_close_pct',
        'momentum_1d', 'momentum_3d', 'momentum_5d',
        'volatility_5d', 'volatility_10d', 'ma_crossover'
    ]
    X = df[feature_columns].values
    y = df['target'].values
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scale target
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
    
    # Data augmentation
    X_aug = np.concatenate([
        X_seq,
        add_noise(X_seq, 0.01),  # Add small noise
        add_noise(X_seq, 0.02)   # Add larger noise
    ])
    y_aug = np.concatenate([y_seq, y_seq, y_seq])
    
    # Shuffle augmented data
    shuffle_idx = np.random.permutation(len(X_aug))
    X_aug = X_aug[shuffle_idx]
    y_aug = y_aug[shuffle_idx]
    
    # Split data
    train_size = int(len(X_aug) * 0.8)
    X_train = X_aug[:train_size]
    X_val = X_aug[train_size:]
    y_train = y_aug[:train_size]
    y_val = y_aug[train_size:]
    
    return X_train, X_val, y_train, y_val, scaler, target_scaler

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, patience=15):
    """Enhanced training function with learning rate scheduling and early stopping."""
    # Initialize optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Convert data to tensors
    X_train = torch.FloatTensor(X_train).to(model.device)
    y_train = torch.FloatTensor(y_train).to(model.device)
    X_val = torch.FloatTensor(X_val).to(model.device)
    y_val = torch.FloatTensor(y_val).to(model.device)
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Create data loaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training loop with progress bar
    for epoch in range(epochs):
        model.model.train()
        train_losses = []
        
        # Training
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for batch_X, batch_y in pbar:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
                
                # Update progress bar
                pbar.set_postfix({'train_loss': f'{np.mean(train_losses):.4f}'})
        
        # Validation
        model.model.eval()
        with torch.no_grad():
            val_outputs = model.model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
            val_loss = val_loss.item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record losses
        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log progress
        logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping check
        if early_stopping(val_loss, model.model):
            logger.info("Early stopping triggered")
            model.model.load_state_dict(early_stopping.best_model)
            break
    
    return history

def main():
    # Change to project root directory
    os.chdir(project_root)
    
    try:
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv('data/processed/test_data.csv')
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_val, y_train, y_val, scaler, target_scaler = prepare_data(df, seq_length=10)
        
        # Initialize model
        logger.info("Initializing model...")
        model = StockPriceLSTM(
            input_size=X_train.shape[2],  # Number of features
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )
        
        # Train model
        logger.info("Training model...")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=100,  # Reduced epochs due to augmented data
            batch_size=64  # Increased batch size for stability
        )
        
        # Save scalers
        logger.info("Saving scalers...")
        os.makedirs('models', exist_ok=True)
        np.save('models/feature_scaler.npy', {
            'scale_': scaler.scale_,
            'min_': scaler.min_,
            'data_min_': scaler.data_min_,
            'data_max_': scaler.data_max_,
            'data_range_': scaler.data_range_
        })
        np.save('models/target_scaler.npy', {
            'scale_': target_scaler.scale_,
            'min_': target_scaler.min_,
            'data_min_': target_scaler.data_min_,
            'data_max_': target_scaler.data_max_,
            'data_range_': target_scaler.data_range_
        })
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 