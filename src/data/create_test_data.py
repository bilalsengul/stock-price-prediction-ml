import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 100

# Generate features
data = {
    'open': np.random.uniform(100, 200, n_samples),
    'high': np.random.uniform(150, 250, n_samples),
    'low': np.random.uniform(90, 180, n_samples),
    'volume': np.random.uniform(1000000, 5000000, n_samples),
    'target': np.random.uniform(120, 220, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/processed/test_data.csv', index=False)
print("Sample test data created successfully!") 