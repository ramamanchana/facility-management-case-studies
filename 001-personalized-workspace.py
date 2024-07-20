import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Sample Data
data = {
    'employee_id': [1, 2, 3, 4, 5],
    'job_role': ['Engineer', 'Manager', 'Analyst', 'Developer', 'Designer'],
    'temperature_pref': [22, 21, 23, 22, 21],
    'light_pref': [300, 320, 280, 310, 305],
    'historical_feedback': [4.5, 3.0, 4.0, 5.0, 3.5]
}

# Create DataFrame
df = pd.DataFrame(data)

# Personalization using Collaborative Filtering
X = df[['temperature_pref', 'light_pref']]
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# Sample Input for Prediction (Employee 3)
employee_id = 3
employee_index = df[df['employee_id'] == employee_id].index[0]
nearest_neighbor = indices[employee_index][1]

print(f"Personalized Settings for Employee {employee_id}:")
print(f"Temperature: {df.iloc[nearest_neighbor]['temperature_pref']}")
print(f"Lighting: {df.iloc[nearest_neighbor]['light_pref']}")

# Plot Personalized Preferences
plt.scatter(df['temperature_pref'], df['light_pref'], color='blue', label='Employees')
plt.scatter(df.iloc[nearest_neighbor]['temperature_pref'], df.iloc[nearest_neighbor]['light_pref'], color='red', label='Personalized', marker='x')
plt.xlabel('Temperature Preference')
plt.ylabel('Lighting Preference')
plt.title('Personalized Workplace Experience')
plt.legend()
plt.show()
