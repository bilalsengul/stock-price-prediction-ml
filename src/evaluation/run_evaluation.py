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

def load_scaler():
    """Load the saved feature scaler."""
    scaler_params = np.load('models/feature_scaler.npy', allow_pickle=True).item()
    scaler = MinMaxScaler()
    scaler.scale_ = scaler_params['scale_']
    scaler.min_ = scaler_params['min_']
    scaler.data_min_ = scaler_params['data_min_']
    scaler.data_max_ = scaler_params['data_max_']
    scaler.data_range_ = scaler_params['data_range_']
    return scaler

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
    
    # Load scaler
    print("\nLoading feature scaler...")
    scaler = load_scaler()
    
    # Prepare input data
    X_test = test_data[['open', 'high', 'low', 'volume']].values
    y_true = test_data['target'].values
    
    # Scale features
    print("\nScaling features...")
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model
    print("\nInitializing model...")
    model = StockPriceLSTM(sequence_length=1, n_features=X_test.shape[1])
    print(f"Using device: {model.device}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test_scaled).flatten()
    
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