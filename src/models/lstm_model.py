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
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size if i == 0 else hidden_size * 2,
                hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        x = self.input_norm(x)
        
        lstm_out = x
        for i in range(self.num_layers):
            residual = lstm_out
            lstm_out, _ = self.lstm_layers[i](lstm_out)
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.dropouts[i](lstm_out)
            if i > 0:  # Add residual connection after first layer
                lstm_out = lstm_out + residual
        
        attention_weights = self.attention(lstm_out)
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        context = context.squeeze(1)
        
        out = self.fc_layers(context)
        return out

class StockPriceLSTM:
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        """Initialize the LSTM model for stock price prediction."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Initialize the model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # First restart after 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model architecture:\n{self.model}")
    
    @staticmethod
    def load_model(filepath=None):
        """Load a pretrained model or create a new one for testing."""
        model = StockPriceLSTM(input_size=7)  # 7 features: open, high, low, volume, percentage change, etc.
        if filepath and os.path.exists(filepath):
            model.model.load_state_dict(torch.load(filepath))
        return model
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 32) -> dict:
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
            patience = 15  # Increased patience for early stopping
            no_improve_count = 0
            
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
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                
                # Step the scheduler
                self.scheduler.step()
                
                # Save best model and check early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), "models/best_model.pth")
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if no_improve_count >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
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
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted values as a 1D array of shape (n_samples,)
        """
        try:
            self.model.eval()
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.squeeze()  # Remove any extra dimensions
            
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
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

def main():
    """
    Main function to demonstrate the usage of StockPriceLSTM.
    """
    try:
        # Example usage (assuming data is prepared)
        sequence_length = 60
        n_features = 8  # OHLCV + percentage change
        
        # Initialize model
        model = StockPriceLSTM(input_size=n_features)
        
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