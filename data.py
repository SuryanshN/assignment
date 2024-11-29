import pandas as pd
import numpy as np
# from sklearn.ensemble import IsolationForest
import os

# Settings
num_samples = 10160340  # 1 GB of data estimated (19,160,340 rows)
anomaly_probability = 0.01  # 10% chance of anomalies

# Generate Synthetic Data
np.random.seed(42)
timestamps = pd.date_range(start="2024-01-01", periods=num_samples, freq="S")
cpu_temperatures = np.random.normal(60, 5, num_samples)
cpu_usages = np.random.normal(50, 15, num_samples)
cpu_loads = np.random.normal(2, 0.5, num_samples)
memory_usages = np.random.normal(50, 10, num_samples)
battery_levels = np.random.uniform(10, 100, num_samples)
cpu_powers = np.random.normal(20, 5, num_samples)

# Introduce anomalies
anomaly_indices = np.random.choice(
    num_samples, int(num_samples * anomaly_probability), replace=False
)
cpu_temperatures[anomaly_indices] += np.random.uniform(30, 50, len(anomaly_indices))
cpu_usages[anomaly_indices] += np.random.uniform(40, 50, len(anomaly_indices))
memory_usages[anomaly_indices] += np.random.uniform(30, 40, len(anomaly_indices))
battery_levels[anomaly_indices] -= np.random.uniform(10, 30, len(anomaly_indices))
cpu_powers[anomaly_indices] += np.random.uniform(30, 50, len(anomaly_indices))

# Create DataFrame
data = {
    "timestamp": timestamps,
    "cpu_temperature": cpu_temperatures,
    "cpu_usage": cpu_usages,
    "cpu_load": cpu_loads,
    "memory_usage": memory_usages,
    "battery_level": battery_levels,
    "cpu_power": cpu_powers,
}
df = pd.DataFrame(data)

# Save to CSV (for large data, use chunks to avoid memory issues)
output_file = "synthetic_hardware_monitor_data.csv"
df.to_csv(output_file, index=False)


