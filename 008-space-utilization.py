import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample Data
data = {
    'room_id': [1, 2, 3, 4, 5],
    'occupancy': [80, 85, 75, 90, 95],
    'room_bookings': [10, 12, 8, 15, 13],
    'employee_schedules': [5, 6, 4, 7, 6]
}

# Create DataFrame
df = pd.DataFrame(data)

# K-Means Clustering for Space Utilization Analysis
X = df[['occupancy', 'room_bookings', 'employee_schedules']]
kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

# Sample Input for Prediction
new_data = pd.DataFrame({
    'occupancy': [88],
    'room_bookings': [11],
    'employee_schedules': [5]
})

predicted_cluster = kmeans.predict(new_data)
print("Predicted Cluster for Space Allocation:", predicted_cluster[0])

# Plot Space Utilization
plt.scatter(df['occupancy'], df['room_bookings'], c=df['cluster'], cmap='viridis')
plt.scatter(new_data['occupancy'], new_data['room_bookings'], c='red', label='New Data', marker='x')
plt.xlabel('Occupancy')
plt.ylabel('Room Bookings')
plt.title('Space Utilization Analysis')
plt.legend()
plt.show()
