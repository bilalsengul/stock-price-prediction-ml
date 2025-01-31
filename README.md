# Stock Price Prediction using Machine Learning

This project implements a machine learning model for predicting stock price movements using historical financial data. The implementation includes data preprocessing, feature engineering, model development, and performance evaluation.

## Project Structure

```
├── data/                      # Directory for storing data
├── models/                    # Directory for saving trained models
├── notebooks/                 # Jupyter notebooks for analysis and visualization
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Data loading utilities
│   │   └── preprocessor.py   # Data preprocessing utilities
│   ├── features/             # Feature engineering
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   ├── models/               # Model implementation
│   │   ├── __init__.py
│   │   └── lstm_model.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── evaluation.py
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-price-prediction-ml
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Collection and Preprocessing:
   ```bash
   python src/data/data_loader.py
   ```

2. Feature Engineering and Model Training:
   ```bash
   python src/models/lstm_model.py
   ```

3. For detailed analysis and visualization, explore the Jupyter notebooks in the `notebooks/` directory.

## Model Architecture

The project implements an LSTM (Long Short-Term Memory) neural network for time series prediction. The model architecture includes:
- Multiple LSTM layers for capturing temporal dependencies
- Dropout layers for preventing overfitting
- Dense layers for final predictions

## Feature Engineering

The following technical indicators are implemented:
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Simple Moving Averages (SMA)
- Exponential Moving Averages (EMA)

## Evaluation Metrics

The model's performance is evaluated using:
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) score

## Results

Detailed evaluation results and analysis can be found in the notebooks directory.

## Dependencies

Major dependencies include:
- numpy
- pandas
- scikit-learn
- tensorflow
- yfinance
- ta (Technical Analysis Library)

For a complete list of dependencies, see `requirements.txt`.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 