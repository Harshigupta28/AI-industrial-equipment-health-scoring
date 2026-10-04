import pandas as pd
import numpy as np

# Reproducible simulated sensor data
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "temperature": np.random.uniform(40, 100, n),
    "vibration": np.random.uniform(1, 12, n),
    "pressure": np.random.uniform(2, 10, n),
    "current": np.random.uniform(5, 25, n),
    "rpm": np.random.uniform(800, 2000, n)
})

# Demonstration labels based on simple rules
def classify_machine(row):
    if row["temperature"] > 85 or row["vibration"] > 9:
        return "Critical"
    elif row["temperature"] > 70 or row["vibration"] > 6:
        return "Warning"
    else:
        return "Healthy"

data["status"] = data.apply(classify_machine, axis=1)

# Save the dataset inside the data folder
data.to_csv("data/sensor_data.csv", index=False)

print("Dataset created successfully!")
print(data.head())
print("\nMachine status counts:")
print(data["status"].value_counts())