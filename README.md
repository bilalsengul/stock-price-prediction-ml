# Stock Price Prediction using Machine Learning

This project implements a deep learning model for predicting stock price movements using historical financial data. The implementation uses PyTorch's LSTM architecture to capture temporal dependencies in stock price data, combined with comprehensive data preprocessing and feature engineering.

## Overview

The system is designed to:
- Process historical stock price data (OHLCV: Open, High, Low, Close, Volume)
- Scale and normalize features for optimal model performance
- Train a deep LSTM model for price prediction
- Evaluate model performance using multiple metrics
- Visualize predictions and error distributions

## Project Structure

```
├── data/                      # Directory for storing data
│   ├── raw/                  # Raw data files
│   │   └── stock_data.csv   # Original stock price data
│   └── processed/            # Processed data files
│       └── test_data.csv    # Processed and scaled data
├── models/                    # Directory for saving trained models
│   ├── best_model.pth       # Best trained model weights
│   └── feature_scaler.npy   # Saved feature scaler
├── outputs/                   # Model outputs and evaluation results
│   └── evaluation/          # Evaluation metrics and plots
│       ├── predictions.png  # Actual vs Predicted plot
│       ├── error_dist.png  # Error distribution plot
│       └── metrics.txt     # Detailed metrics
├── src/                      # Source code
│   ├── data/                # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py  # Data loading utilities
│   │   └── preprocessor.py # Data preprocessing utilities
│   ├── models/              # Model implementation
│   │   ├── __init__.py
│   │   ├── lstm_model.py   # LSTM model implementation
│   │   └── train_model.py  # Model training script
│   └── evaluation/          # Evaluation utilities
│       ├── __init__.py
│       └── run_evaluation.py
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Data Processing

### Data Loading
- Supports loading data from CSV files
- Handles missing values through forward/backward filling
- Implements data validation and integrity checks

### Preprocessing Steps
1. **Data Cleaning**:
   - Remove missing values
   - Handle outliers using IQR method
   - Remove duplicate entries

2. **Feature Scaling**:
   - MinMax scaling for all features (0-1 range)
   - Separate scalers for price and volume data
   - Scaler persistence for consistent preprocessing

3. **Sequence Creation**:
   - Creation of time series sequences
   - Configurable sequence length (default: 60 time steps)
   - Sliding window approach for sample generation

## Model Architecture

### LSTM Network Details
```
LSTMModel(
  (lstm): LSTM(
    input_size=4,           # 4 features: open, high, low, volume
    hidden_size=64,         # 64 hidden units
    num_layers=2,           # 2 stacked LSTM layers
    batch_first=True,       # (batch_size, seq_len, features)
    dropout=0.2            # 20% dropout for regularization
  )
  (fc): Linear(
    in_features=64,        # Match LSTM hidden size
    out_features=1         # Single price prediction
  )
)
```

### Key Components:

1. **Input Layer**:
   - Accepts 4 features (OHLV data)
   - Handles both batched and single sample inputs
   - Automatic shape adjustment for sequence dimension

2. **LSTM Layers**:
   - 2 stacked LSTM layers for deep temporal learning
   - 64 hidden units per layer for complex pattern capture
   - Dropout (0.2) between layers for regularization
   - Batch-first configuration for intuitive data handling

3. **Output Layer**:
   - Dense layer for final prediction
   - Single unit output for price prediction
   - Linear activation for regression task

## Training Process

### Configuration
```python
optimizer = Adam(
    lr=0.001,              # Learning rate
    betas=(0.9, 0.999),    # Adam optimizer parameters
    eps=1e-8               # Numerical stability
)
criterion = MSELoss()      # Mean Squared Error loss
batch_size = 32           # Mini-batch size
epochs = 100              # Training epochs
```

### Training Protocol
1. **Data Preparation**:
   - 80/20 train/validation split
   - MinMax scaling of features
   - Batch creation with DataLoader

2. **Training Loop**:
   - Forward pass through LSTM
   - Loss calculation using MSE
   - Backpropagation and optimization
   - Validation after each epoch
   - Model checkpointing for best validation loss

3. **Monitoring**:
   - Training loss tracking
   - Validation loss monitoring
   - Learning rate adjustment
   - Early stopping capability

### Training Progress
```
Epoch [10/100], Train Loss: 30694.3151, Val Loss: 28954.2754
Epoch [20/100], Train Loss: 29198.5143, Val Loss: 28522.6387
Epoch [30/100], Train Loss: 28208.7285, Val Loss: 27659.2441
Epoch [40/100], Train Loss: 27959.5228, Val Loss: 26938.1309
Epoch [50/100], Train Loss: 26983.7812, Val Loss: 26348.6250
Epoch [60/100], Train Loss: 26400.6224, Val Loss: 25805.9277
Epoch [70/100], Train Loss: 24729.6087, Val Loss: 25291.1484
Epoch [80/100], Train Loss: 24790.5540, Val Loss: 24798.2168
Epoch [90/100], Train Loss: 24725.5345, Val Loss: 24323.1543
Epoch [100/100], Train Loss: 24111.4486, Val Loss: 23856.3730
```

## Latest Evaluation Results

### Performance Metrics
| Metric | Value | Description |
|--------|--------|-------------|
| MSE    | 30474.36 | Mean Squared Error |
| RMSE   | 174.57 | Root Mean Squared Error |
| MAE    | 171.67 | Mean Absolute Error |
| R²     | -29.33 | Coefficient of Determination |
| MAPE   | 100.04% | Mean Absolute Percentage Error |

### Error Analysis
- High RMSE indicates significant prediction errors
- Negative R² suggests model underperformance
- High MAPE indicates poor percentage accuracy

### Visualization Outputs
1. **Predictions Plot** (`predictions.png`):
   - Time series of actual vs. predicted values
   - Confidence intervals for predictions
   - Trend analysis visualization

2. **Error Distribution** (`error_distribution.png`):
   - Histogram of prediction errors
   - Normal distribution overlay
   - Error statistics summary

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- 8GB RAM minimum

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-price-prediction-ml
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables
Create a `.env` file with the following variables:
```env
CUDA_VISIBLE_DEVICES=0  # GPU device index
SEED=42                # Random seed for reproducibility
LOG_LEVEL=INFO        # Logging level
```

## Usage

### Data Preparation
```bash
# Download and prepare stock data
python src/data/data_loader.py --symbol AAPL --start 2010-01-01

# Process and create features
python src/data/preprocessor.py --input data/raw/stock_data.csv
```

### Model Training
```bash
# Train the model
python src/models/train_model.py --epochs 100 --batch-size 32

# Resume training from checkpoint
python src/models/train_model.py --resume --checkpoint models/best_model.pth
```

### Model Evaluation
```bash
# Evaluate the model
python src/evaluation/run_evaluation.py --model models/best_model.pth
```

## Dependencies

### Core Libraries
- PyTorch >= 2.2.0 (Deep Learning)
- NumPy >= 1.21.0 (Numerical Computing)
- Pandas >= 1.3.0 (Data Manipulation)
- Scikit-learn >= 0.24.2 (Machine Learning)
- Matplotlib >= 3.4.3 (Plotting)
- Seaborn >= 0.11.2 (Statistical Visualization)

### Additional Requirements
- yfinance >= 0.1.63 (Yahoo Finance API)
- python-dotenv >= 0.19.0 (Environment Variables)
- tqdm >= 4.62.3 (Progress Bars)

## Future Improvements

### 1. Data Collection
- Implement real-time data fetching using yfinance
- Add support for multiple data sources:
  - Alpha Vantage API
  - Quandl
  - Bloomberg API
- Include more financial indicators:
  - Order book data
  - Market sentiment
  - Economic indicators

### 2. Model Enhancement
- Implement attention mechanism:
  - Self-attention layers
  - Multi-head attention
- Add support for multi-step prediction:
  - Sequence-to-sequence architecture
  - Teacher forcing
- Experiment with architectures:
  - GRU (Gated Recurrent Unit)
  - Transformer
  - Temporal Convolutional Networks

### 3. Feature Engineering
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- Sentiment analysis:
  - News headlines
  - Social media sentiment
  - Market reports
- Feature selection:
  - Recursive feature elimination
  - LASSO regularization
  - Principal Component Analysis

### 4. Training Optimization
- Early stopping:
  - Patience-based stopping
  - Loss plateau detection
- Learning rate scheduling:
  - Cyclic learning rates
  - Cosine annealing
  - Warm restarts
- Cross-validation:
  - Time series cross-validation
  - Walk-forward optimization
- Hyperparameter tuning:
  - Optuna optimization
  - Ray Tune integration
  - Bayesian optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the PyTorch team for the excellent deep learning framework
- Yahoo Finance for providing financial data
- The open-source community for various tools and libraries 