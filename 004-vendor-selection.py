import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample Data
data = {
    'vendor_id': [1, 2, 3, 4, 5],
    'delivery_timelines': [10, 15, 9, 12, 14],
    'quality_scores': [4.5, 3.0, 4.8, 4.0, 3.5],
    'cost_data': [1000, 1500, 1100, 1200, 1400],
    'contract_terms': [2, 1, 3, 2, 1],
    'reliability': [1, 0, 1, 1, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Random Forest Model for Vendor Performance Analysis
X = df[['delivery_timelines', 'quality_scores', 'cost_data', 'contract_terms']]
y = df['reliability']

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'delivery_timelines': [11],
    'quality_scores': [4.2],
    'cost_data': [1150],
    'contract_terms': [2]
})

# Prediction
predicted_reliability = model.predict(new_data)
print("Predicted Vendor Reliability:", predicted_reliability[0])

# Plot Vendor Performance
plt.scatter(df['vendor_id'], df['reliability'], color='blue', label='Actual')
plt.scatter([6], predicted_reliability, color='red', label='Predicted', marker='x')
plt.xlabel('Vendor ID')
plt.ylabel('Reliability')
plt.title('Vendor Performance Analysis')
plt.legend()
plt.show()
