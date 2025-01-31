import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = LSTMModel(input_size=n_features).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model architecture:\n{self.model}")
        
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
            # Create data loaders
            train_dataset = TimeSeriesDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            history = {'train_loss': [], 'val_loss': []}
            best_val_loss = float('inf')
            
            # Create directory for model checkpoints
            os.makedirs("models", exist_ok=True)
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_losses = []
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y.unsqueeze(1))
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Validation
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y.unsqueeze(1))
                        val_losses.append(loss.item())
                
                # Calculate average losses
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "models/best_model.pth")
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Load best model
            self.model.load_state_dict(torch.load("models/best_model.pth"))
            logger.info("Model training completed successfully")
            return history
            
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
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.cpu().numpy()
            
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
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'sequence_length': self.sequence_length,
                'n_features': self.n_features
            }, filepath)
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
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.sequence_length = checkpoint['sequence_length']
            self.n_features = checkpoint['n_features']
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
        
        # Generate sample data for testing
        X_train = np.random.random((1000, sequence_length, n_features))
        y_train = np.random.random(1000)
        X_val = np.random.random((200, sequence_length, n_features))
        y_val = np.random.random(200)
        
        # Train model
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model.save_model('models/final_model.pth')
        
        logger.info("Model training and saving completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 