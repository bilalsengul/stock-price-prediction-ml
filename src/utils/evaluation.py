import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        """
        Initialize the ModelEvaluator class.
        """
        pass
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate various performance metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            dict: Dictionary containing various performance metrics
        """
        try:
            # Ensure arrays are flattened
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
            
            logger.info("Performance metrics calculated:")
            for metric, value in metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
            
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Actual vs Predicted Values",
                        save_path: str = None):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
            plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            raise
            
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Prediction Error Distribution",
                              save_path: str = None):
        """
        Plot the distribution of prediction errors.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        try:
            errors = y_true - y_pred
            
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.title(title)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting error distribution: {str(e)}")
            raise
            
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 output_dir: str = "reports"):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            output_dir (str): Directory to save the report and plots
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred)
            
            # Generate plots
            self.plot_predictions(y_true, y_pred,
                                save_path=f"{output_dir}/predictions_plot.png")
            self.plot_error_distribution(y_true, y_pred,
                                       save_path=f"{output_dir}/error_distribution.png")
            
            # Create markdown report
            report = f"""# Model Evaluation Report

## Performance Metrics

- MSE: {metrics['mse']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- MAPE: {metrics['mape']:.4f}%
- R²: {metrics['r2']:.4f}

## Visualization

### Actual vs Predicted Values
![Predictions Plot](predictions_plot.png)

### Error Distribution
![Error Distribution](error_distribution.png)

## Analysis

The model shows the following characteristics:

1. Overall Performance:
   - The R² score of {metrics['r2']:.4f} indicates {'good' if metrics['r2'] > 0.7 else 'moderate' if metrics['r2'] > 0.5 else 'poor'} predictive power
   - MAPE of {metrics['mape']:.2f}% suggests {'excellent' if metrics['mape'] < 10 else 'good' if metrics['mape'] < 20 else 'moderate' if metrics['mape'] < 30 else 'poor'} prediction accuracy

2. Error Analysis:
   - RMSE: {metrics['rmse']:.4f}
   - MAE: {metrics['mae']:.4f}
   - The difference between RMSE and MAE indicates the presence of {'significant' if metrics['rmse'] - metrics['mae'] > metrics['mae']*0.2 else 'moderate' if metrics['rmse'] - metrics['mae'] > metrics['mae']*0.1 else 'minimal'} outliers in predictions
"""
            
            # Save report
            with open(f"{output_dir}/evaluation_report.md", 'w') as f:
                f.write(report)
                
            logger.info(f"Evaluation report generated in {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of ModelEvaluator.
    """
    try:
        # Generate sample data
        np.random.seed(42)
        y_true = np.random.normal(100, 10, 100)
        y_pred = y_true + np.random.normal(0, 5, 100)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Generate evaluation report
        evaluator.generate_evaluation_report(y_true, y_pred)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 