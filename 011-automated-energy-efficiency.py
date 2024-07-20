import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample Data
data = {
    'timestamp': pd.date_range(start='1/1/2022', periods=100, freq='H'),
    'energy_usage': np.random.rand(100) * 100,
    'occupancy': np.random.randint(1, 100, size=100),
    'equipment_usage': np.random.randint(1, 100, size=100)
}

# Create DataFrame
df = pd.DataFrame(data)

# Normalize the dataset
df['energy_usage'] = df['energy_usage'] / df['energy_usage'].max()
df['occupancy'] = df['occupancy'] / df['occupancy'].max()
df['equipment_usage'] = df['equipment_usage'] / df['equipment_usage'].max()

# Create input and output datasets
X = df[['occupancy', 'equipment_usage']].values
Y = df['energy_usage'].values

# Neural Network Model
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train Model
model.fit(X, Y, epochs=10, batch_size=2, verbose=2)

# Make predictions
predictions = model.predict(X)

# Plot Energy Usage and Predictions
plt.plot(df['timestamp'], df['energy_usage'], label='Actual')
plt.plot(df['timestamp'], predictions, label='Predicted', linestyle='--')
plt.xlabel('Timestamp')
plt.ylabel('Energy Usage')
plt.title('Automated Energy Efficiency Optimization')
plt.legend()
plt.show()
