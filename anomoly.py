import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("synthetic_hardware_monitor_data.csv")
# Initialize Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
features = ['cpu_temperature', 'cpu_usage', 'cpu_load', 'memory_usage', 'cpu_power']

# Fit and predict anomalies
df['anomaly'] = iso_forest.fit_predict(df[features])
df['anomaly'] = df['anomaly'].apply(lambda x: x == -1)
# Plot each feature
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].plot(df.index, df[feature], label='Data')
    axes[i].scatter(
        df[df['anomaly']].index,
        df[df['anomaly']][feature],
        color='red',
        label='Anomalies'
    )
    axes[i].set_title(f'{feature.capitalize()} with Anomalies Highlighted')
    axes[i].set_xlabel('Index')
    axes[i].set_ylabel(feature.capitalize())
    axes[i].legend()

plt.tight_layout()
plt.show()
