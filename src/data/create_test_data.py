import numpy as np
import pandas as pd

def generate_realistic_stock_data(n_samples=100, volatility=0.02, trend=0.001):
    """Generate realistic stock price data with trends and patterns."""
    # Initial price
    initial_price = 150
    
    # Generate daily returns with a slight upward trend and volatility
    returns = np.random.normal(trend, volatility, n_samples)
    
    # Calculate prices using cumulative returns
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate realistic OHLC data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'close': prices,
        'high': prices * (1 + abs(np.random.normal(0, 0.004, n_samples))),
        'low': prices * (1 - abs(np.random.normal(0, 0.004, n_samples))),
        'volume': np.random.lognormal(15, 1, n_samples)
    }
    
    # Ensure high is highest and low is lowest
    data['high'] = np.maximum(
        np.maximum(data['high'], data['open']),
        data['close']
    )
    data['low'] = np.minimum(
        np.minimum(data['low'], data['open']),
        data['close']
    )
    
    # Create target (next day's closing price)
    data['target'] = np.roll(data['close'], -1)
    data['target'][-1] = data['close'][-1] * (1 + returns[-1])
    
    return pd.DataFrame(data)

# Generate sample data
np.random.seed(42)
df = generate_realistic_stock_data(n_samples=1000)  # Increased sample size

# Save to CSV
df.to_csv('data/processed/test_data.csv', index=False)
print("Realistic test data created successfully!")
print(f"Data shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData statistics:")
print(df.describe()) 