import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Sample Data
data = {
    'equipment_id': [1, 2, 3, 4, 5],
    'temperature': [70, 75, 68, 72, 74],
    'vibration': [0.4, 0.5, 0.3, 0.45, 0.5],
    'pressure': [30, 35, 28, 33, 34],
    'maintenance_history': [2, 3, 1, 2, 3],
    'failures': [1, 2, 0, 1, 2]
}

# Create DataFrame
df = pd.DataFrame(data)

# Random Forest Model for Predictive Maintenance
X = df[['temperature', 'vibration', 'pressure', 'maintenance_history']]
y = df['failures']

# Train Model
model = RandomForestRegressor()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'temperature': [73],
    'vibration': [0.47],
    'pressure': [32],
    'maintenance_history': [2]
})

# Prediction
predicted_failure = model.predict(new_data)
print("Predicted Equipment Failure:", predicted_failure[0])

# Plot Maintenance Data
plt.scatter(df['equipment_id'], df['failures'], color='blue', label='Actual')
plt.scatter([6], predicted_failure, color='red', label='Predicted', marker='x')
plt.xlabel('Equipment ID')
plt.ylabel('Failures')
plt.title('Predictive Maintenance for Equipment')
plt.legend()
plt.show()
