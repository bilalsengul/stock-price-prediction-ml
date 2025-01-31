import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.models.lstm_model import StockPriceLSTM
from src.evaluation.model_evaluator import ModelEvaluator

def main():
    # Change working directory to project root
    os.chdir(project_root)
    
    # Set up plotting style
    plt.style.use('seaborn')
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv('data/processed/test_data.csv')
    
    # Convert data to tensor format
    X_test = torch.FloatTensor(test_data.drop('target', axis=1).values)
    y_test = torch.FloatTensor(test_data['target'].values)
    
    print("\nDataset Info:")
    print(test_data.info())
    print("\nFirst few rows:")
    print(test_data.head())
    
    # Load and evaluate model
    print("\nLoading model...")
    model = StockPriceLSTM.load_model('models/final_model.pth')
    model.eval()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_true, y_pred = evaluator.get_predictions(X_test)
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("\nModel Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric:15s}: {value:.4f}")
    
    # Create output directory for plots
    os.makedirs('outputs/evaluation', exist_ok=True)
    
    # Generate and save plots
    print("\nGenerating visualizations...")
    
    # Plot predictions vs actual values
    evaluator.plot_predictions(y_true, y_pred)
    plt.savefig('outputs/evaluation/predictions_vs_actual.png')
    plt.close()
    
    # Plot error distribution
    evaluator.plot_error_distribution(y_true, y_pred)
    plt.savefig('outputs/evaluation/error_distribution.png')
    plt.close()
    
    # Generate comprehensive report
    print("\nGenerating evaluation report...")
    report = evaluator.generate_evaluation_report(y_true, y_pred)
    
    # Save report to file
    with open('outputs/evaluation/evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("\nEvaluation complete! Results saved in outputs/evaluation/")
    print("- predictions_vs_actual.png")
    print("- error_distribution.png")
    print("- evaluation_report.txt")

if __name__ == "__main__":
    main() 