import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPriceLSTM:
    def __init__(self, sequence_length: int = 60, n_features: int = 5):
        """
        Initialize the LSTM model for stock price prediction.
        
        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of features in the input data
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self) -> Sequential:
        """
        Build and compile the LSTM model.
        
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error')
        
        logger.info(model.summary())
        return model
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        try:
            # Create model checkpoint callback
            checkpoint_path = "models/best_model.h5"
            os.makedirs("models", exist_ok=True)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                              save_best_only=True, mode='min')
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return history.history
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        try:
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred.flatten()) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test - y_pred.flatten()))
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            logger.info("Model evaluation completed:")
            for metric, value in metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
            
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            self.model.save(filepath)
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded successfully from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of StockPriceLSTM.
    """
    try:
        # Example usage (assuming data is prepared)
        sequence_length = 60
        n_features = 5  # OHLCV
        
        # Initialize model
        model = StockPriceLSTM(sequence_length=sequence_length,
                              n_features=n_features)
        
        # Load and prepare data (placeholder)
        # In practice, this would come from your data preparation pipeline
        X_train = np.random.random((1000, sequence_length, n_features))
        y_train = np.random.random(1000)
        X_val = np.random.random((200, sequence_length, n_features))
        y_val = np.random.random(200)
        
        # Train model
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model.save_model('models/final_model.h5')
        
        logger.info("Model training and saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 