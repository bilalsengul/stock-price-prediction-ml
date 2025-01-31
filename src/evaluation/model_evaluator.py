import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        
    def get_predictions(self, test_data):
        """Generate predictions using the model."""
        try:
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                predictions = self.model(test_data)
            return test_data['target'].values, predictions.numpy()
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate various performance metrics."""
        try:
            metrics = {
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            logger.info("Metrics calculated successfully")
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def plot_predictions(self, y_true, y_pred):
        """Plot actual vs predicted values."""
        try:
            plt.figure(figsize=(15, 6))
            plt.plot(y_true, label='Actual', color='blue', alpha=0.7)
            plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
            plt.title('Stock Price: Actual vs Predicted Values')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.grid(True)
            logger.info("Prediction plot generated successfully")
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            raise
    
    def plot_error_distribution(self, y_true, y_pred):
        """Plot the distribution of prediction errors."""
        try:
            errors = y_true - y_pred
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, kde=True)
            plt.title('Distribution of Prediction Errors')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            logger.info("Error distribution plot generated successfully")
        except Exception as e:
            logger.error(f"Error plotting error distribution: {str(e)}")
            raise
    
    def generate_evaluation_report(self, y_true, y_pred):
        """Generate a comprehensive evaluation report."""
        try:
            metrics = self.calculate_metrics(y_true, y_pred)
            
            report = "Model Evaluation Report\n"
            report += "=" * 50 + "\n\n"
            
            # Performance metrics
            report += "Performance Metrics:\n"
            report += "-" * 20 + "\n"
            for metric, value in metrics.items():
                report += f"{metric:15s}: {value:.4f}\n"
            
            # Error analysis
            errors = y_true - y_pred
            report += "\nError Analysis:\n"
            report += "-" * 20 + "\n"
            report += f"Mean Error: {np.mean(errors):.4f}\n"
            report += f"Std Error: {np.std(errors):.4f}\n"
            report += f"Min Error: {np.min(errors):.4f}\n"
            report += f"Max Error: {np.max(errors):.4f}\n"
            
            # Directional accuracy
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
            report += f"\nDirectional Accuracy: {directional_accuracy:.2f}%\n"
            
            logger.info("Evaluation report generated successfully")
            return report
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            raise 