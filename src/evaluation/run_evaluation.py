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

from src.models.lstm_model import StockPriceLSTM

def calculate_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def main():
    # Change to project root directory
    os.chdir(project_root)
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Prepare input data
    X_test = torch.FloatTensor(test_data.drop('target', axis=1).values)
    y_true = test_data['target'].values
    
    # Load model
    print("\nLoading model...")
    model = StockPriceLSTM.load_model()
    model.eval()
    
    # Generate predictions
    print("\nGenerating predictions...")
    with torch.no_grad():
        y_pred = model(X_test).numpy().flatten()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)
    
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    # Create plots directory
    os.makedirs('outputs/evaluation', exist_ok=True)
    
    # Plot predictions vs actual
    print("\nGenerating plots...")
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title('Stock Price: Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/evaluation/predictions.png')
    plt.close()
    
    print("\nEvaluation complete! Results saved in outputs/evaluation/")

if __name__ == "__main__":
    main() 