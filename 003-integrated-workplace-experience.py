import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample Data
data = {
    'occupancy': [80, 85, 75, 90, 95],
    'temperature': [22, 21, 23, 22, 21],
    'humidity': [40, 45, 35, 50, 42],
    'air_quality': [80, 75, 85, 70, 90],
    'energy_usage': [200, 220, 210, 230, 240]
}

# Create DataFrame
df = pd.DataFrame(data)

# Gradient Boosting for Optimal Workplace Conditions
X = df[['occupancy', 'temperature', 'humidity', 'air_quality']]
y = df['energy_usage']

# Train Model
model = GradientBoostingRegressor()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'occupancy': [88],
    'temperature': [22],
    'humidity': [43],
    'air_quality': [78]
})

# Prediction
predicted_energy_usage = model.predict(new_data)
print("Predicted Energy Usage:", predicted_energy_usage[0])

# Plot Energy Usage
plt.scatter(df['occupancy'], df['energy_usage'], color='blue', label='Actual')
plt.scatter(new_data['occupancy'], predicted_energy_usage, color='red', label='Predicted')
plt.xlabel('Occupancy')
plt.ylabel('Energy Usage')
plt.title('Technologically Integrated Workplace Experience')
plt.legend()
plt.show()
