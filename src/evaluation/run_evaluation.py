import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.models.lstm_model import StockPriceLSTM
from src.models.train_model import prepare_price_features, create_sequences

def load_scalers():
    """Load the saved feature and target scalers."""
    # Load feature scaler
    feature_scaler_params = np.load('models/feature_scaler.npy', allow_pickle=True).item()
    feature_scaler = MinMaxScaler()
    feature_scaler.scale_ = feature_scaler_params['scale_']
    feature_scaler.min_ = feature_scaler_params['min_']
    feature_scaler.data_min_ = feature_scaler_params['data_min_']
    feature_scaler.data_max_ = feature_scaler_params['data_max_']
    feature_scaler.data_range_ = feature_scaler_params['data_range_']
    
    # Load target scaler
    target_scaler_params = np.load('models/target_scaler.npy', allow_pickle=True).item()
    target_scaler = MinMaxScaler()
    target_scaler.scale_ = target_scaler_params['scale_']
    target_scaler.min_ = target_scaler_params['min_']
    target_scaler.data_min_ = target_scaler_params['data_min_']
    target_scaler.data_max_ = target_scaler_params['data_max_']
    target_scaler.data_range_ = target_scaler_params['data_range_']
    
    return feature_scaler, target_scaler

def calculate_metrics(y_true, y_pred):
    """Calculate various performance metrics."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

def plot_predictions(y_true, y_pred, save_path):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title('Stock Price: Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(y_true, y_pred, save_path):
    """Plot error distribution."""
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Change to project root directory
    os.chdir(project_root)
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Prepare features
    print("\nPreparing features...")
    test_data = prepare_price_features(test_data)
    
    # Load scalers
    print("\nLoading scalers...")
    feature_scaler, target_scaler = load_scalers()
    
    # Prepare input data
    feature_columns = [
        'return_1d', 'high_low_pct', 'open_close_pct',
        'momentum_1d', 'momentum_3d', 'momentum_5d',
        'volatility_5d', 'volatility_10d', 'ma_crossover'
    ]
    X_test = test_data[feature_columns].values
    y_true = test_data['target'].values
    
    # Scale features and target
    print("\nScaling data...")
    X_test_scaled = feature_scaler.transform(X_test)
    y_true_scaled = target_scaler.transform(y_true.reshape(-1, 1)).flatten()
    
    # Create sequences
    print("\nCreating sequences...")
    seq_length = 10
    X_test_seq, y_true_seq = create_sequences(X_test_scaled, y_true_scaled, seq_length)
    
    # Initialize model
    print("\nInitializing model...")
    model = StockPriceLSTM(
        input_size=X_test_seq.shape[2],  # Number of features
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    print(f"Using device: {model.device}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_scaled = model.predict(X_test_seq).flatten()
    
    # Inverse transform predictions
    print("\nInverse transforming predictions...")
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_true[seq_length:]  # Align with predictions
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    # Create output directory
    os.makedirs('outputs/evaluation', exist_ok=True)
    
    # Generate and save plots
    print("\nGenerating visualizations...")
    plot_predictions(y_true, y_pred, 'outputs/evaluation/predictions.png')
    plot_error_distribution(y_true, y_pred, 'outputs/evaluation/error_distribution.png')
    
    # Save metrics to file
    with open('outputs/evaluation/metrics.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
    
    print("\nEvaluation complete! Results saved in outputs/evaluation/")
    print("- predictions.png")
    print("- error_distribution.png")
    print("- metrics.txt")

if __name__ == "__main__":
    main() 