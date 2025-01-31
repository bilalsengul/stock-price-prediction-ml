# Stock Price Prediction using Machine Learning

This project implements a machine learning model for predicting stock price movements using historical financial data. The implementation includes data preprocessing, feature engineering, model development, and performance evaluation.

## Project Structure

```
├── data/                      # Directory for storing data
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── models/                    # Directory for saving trained models
│   ├── best_model.pth       # Best trained model weights
│   └── feature_scaler.npy   # Saved feature scaler
├── outputs/                   # Model outputs and evaluation results
│   └── evaluation/          # Evaluation metrics and plots
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

## Model Architecture

The project implements an LSTM (Long Short-Term Memory) neural network for time series prediction. The model architecture includes:

- Input Layer: Accepts 4 features (open, high, low, volume)
- LSTM Layers:
  - 2 stacked LSTM layers with 64 hidden units each
  - Dropout rate of 0.2 for regularization
  - Batch-first configuration for easier data handling
- Output Layer: Dense layer with 1 unit for price prediction

## Training Process

The model is trained using the following configuration:
- Optimizer: Adam with learning rate 0.001
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 32
- Epochs: 100
- Train/Validation Split: 80/20

Training progress shows consistent decrease in both training and validation loss:
```
Epoch [10/100], Train Loss: 30694.3151, Val Loss: 28954.2754
Epoch [20/100], Train Loss: 29198.5143, Val Loss: 28522.6387
Epoch [30/100], Train Loss: 28208.7285, Val Loss: 27659.2441
...
Epoch [90/100], Train Loss: 24725.5345, Val Loss: 24323.1543
Epoch [100/100], Train Loss: 24111.4486, Val Loss: 23856.3730
```

## Latest Evaluation Results

The model's performance on the test dataset:

| Metric | Value |
|--------|--------|
| MSE    | 30474.36 |
| RMSE   | 174.57 |
| MAE    | 171.67 |
| R²     | -29.33 |
| MAPE   | 100.04% |

Note: The current metrics indicate that the model needs further improvement. This could be achieved through:
1. Using real historical stock data instead of randomly generated data
2. Feature engineering to include technical indicators
3. Hyperparameter optimization
4. Increasing the sequence length to capture longer-term patterns

## Setup and Installation

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

## Usage

1. Train the model:
   ```bash
   python src/models/train_model.py
   ```

2. Evaluate the model:
   ```bash
   python src/evaluation/run_evaluation.py
   ```

The evaluation results will be saved in the `outputs/evaluation/` directory:
- `predictions.png`: Plot of actual vs predicted values
- `error_distribution.png`: Distribution of prediction errors
- `metrics.txt`: Detailed performance metrics

## Dependencies

Major dependencies include:
- PyTorch >= 2.2.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.2
- Matplotlib >= 3.4.3
- Seaborn >= 0.11.2

For a complete list of dependencies, see `requirements.txt`.

## Future Improvements

1. Data Collection:
   - Implement real-time data fetching using yfinance
   - Add support for multiple data sources
   - Include more financial indicators

2. Model Enhancement:
   - Implement attention mechanism
   - Add support for multi-step prediction
   - Experiment with different architectures (GRU, Transformer)

3. Feature Engineering:
   - Add technical indicators (RSI, MACD, Bollinger Bands)
   - Include sentiment analysis from news/social media
   - Implement automatic feature selection

4. Training Optimization:
   - Add early stopping
   - Implement learning rate scheduling
   - Add cross-validation
   - Hyperparameter tuning using Optuna/Ray Tune

## License

This project is licensed under the MIT License - see the LICENSE file for details. 